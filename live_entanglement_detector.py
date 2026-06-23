#!/usr/bin/env python3
"""Live entanglement detector for Unitree Go2 LowState.

This script is designed for two modes:

1. Live ROS2 mode
   Subscribe to /lowstate, compute a sliding-window anomaly score per leg,
   and raise warnings when one leg stays suspicious for multiple windows.

2. Replay mode
   Feed a CSV log through the same detector logic without ROS2. This is useful
   for testing and for validating thresholds against normal walking logs.

The detector uses a walking baseline CSV to calibrate per-leg thresholds.
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Deque, Iterable

import pandas as pd


LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_ORDER = ("hip", "thigh", "calf")
DEFAULT_BASELINE_CSV = Path("data/go2_lowstate_walking.csv")
DEFAULT_WINDOW_SECONDS = 0.30
DEFAULT_PERSISTENCE_WINDOWS = 3
DEFAULT_SCORE_PERCENTILE = 99.5
DEFAULT_THRESHOLD_SCALE = 0.80

CSV_FOOT_ORDER = ("FL", "FR", "RL", "RR")
MOTOR_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}

FEATURE_WEIGHTS = {
    "tau_sum": 2.5,
    "low_velocity": 2.0,
    "foot_force": 1.4,
    "q_mean": 0.9,
    "rpy_mean": 0.8,
    "gyro_mag": 0.7,
    "acc_dev": 0.5,
    "tau_dominance": 1.5,
    "foot_dominance": 0.9,
}


@dataclass(frozen=True)
class FeatureStats:
    mean: float
    std: float


@dataclass(frozen=True)
class BaselineStats:
    per_leg: dict[str, dict[str, FeatureStats]]
    score_thresholds: dict[str, float]


@dataclass(frozen=True)
class SampleFeatures:
    tau_sum: float
    dq_mean: float
    q_mean: float
    foot_force: float
    rpy_mean: float | None
    gyro_mag: float | None
    acc_mag: float | None


@dataclass(frozen=True)
class ScoreResult:
    scores: dict[str, float]
    candidate_leg: str | None
    candidate_score: float
    second_score: float
    alarm_leg: str | None
    alarm_score: float
    threshold: float
    streak: int
    is_new_alarm: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live entanglement detector for Unitree Go2 lowstate streams. Uses a "
            "walking baseline to calibrate thresholds and can also replay CSV logs."
        )
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE_CSV,
        help="CSV log of normal walking used to calibrate thresholds.",
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=None,
        help="Replay a CSV log through the same detector instead of subscribing live.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Sliding window length in seconds for smoothing live measurements.",
    )
    parser.add_argument(
        "--persistence-windows",
        type=int,
        default=DEFAULT_PERSISTENCE_WINDOWS,
        help="How many consecutive suspicious windows are required before an alarm.",
    )
    parser.add_argument(
        "--score-percentile",
        type=float,
        default=DEFAULT_SCORE_PERCENTILE,
        help="Baseline score percentile used as the initial leg-specific threshold.",
    )
    parser.add_argument(
        "--dominance-gap",
        type=float,
        default=0.0,
        help="Minimum score gap between the top leg and second-best leg.",
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=DEFAULT_THRESHOLD_SCALE,
        help="Scale baseline thresholds. Lower values make the detector more sensitive.",
    )
    parser.add_argument(
        "--log-normal",
        action="store_true",
        help="Print non-alarm status messages in live ROS2 mode.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="In replay mode, print only final summary lines.",
    )
    parser.add_argument(
        "--foot-force-order",
        type=str,
        default=",".join(CSV_FOOT_ORDER),
        help="Foot-force order in the ROS message, comma-separated labels like FL,FR,RL,RR.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except TypeError:
        return None
    except ValueError:
        return None


def vector_magnitude(values: Iterable[float] | None) -> float | None:
    if values is None:
        return None
    values_list = list(values)
    if not values_list:
        return None
    return math.sqrt(sum(component * component for component in values_list))


def mean_abs(values: Iterable[float] | None) -> float | None:
    if values is None:
        return None
    values_list = [abs(component) for component in values]
    if not values_list:
        return None
    return fmean(values_list)


def get_nested_vector(source: Any, candidates: tuple[tuple[str, ...], ...]) -> list[float] | None:
    for candidate in candidates:
        current = source
        ok = True
        for attribute in candidate:
            if not hasattr(current, attribute):
                ok = False
                break
            current = getattr(current, attribute)
        if not ok:
            continue
        vector = safe_vector(current)
        if vector is not None and len(vector) >= 3:
            return vector[:3]
    return None


def leg_feature_stats(values: list[SampleFeatures]) -> dict[str, FeatureStats]:
    by_name: dict[str, list[float]] = {
        "tau_sum": [],
        "dq_mean": [],
        "q_mean": [],
        "foot_force": [],
        "rpy_mean": [],
        "gyro_mag": [],
        "acc_mag": [],
    }

    for sample in values:
        by_name["tau_sum"].append(sample.tau_sum)
        by_name["dq_mean"].append(sample.dq_mean)
        by_name["q_mean"].append(sample.q_mean)
        by_name["foot_force"].append(sample.foot_force)
        if sample.rpy_mean is not None:
            by_name["rpy_mean"].append(sample.rpy_mean)
        if sample.gyro_mag is not None:
            by_name["gyro_mag"].append(sample.gyro_mag)
        if sample.acc_mag is not None:
            by_name["acc_mag"].append(sample.acc_mag)

    stats: dict[str, FeatureStats] = {}
    for feature_name, series in by_name.items():
        if not series:
            continue
        mean_value = fmean(series)
        variance = fmean([(item - mean_value) ** 2 for item in series])
        stats[feature_name] = FeatureStats(mean=mean_value, std=math.sqrt(variance) if variance > 0 else 1e-6)
    return stats


def csv_row_features(row: pd.Series, leg: str) -> SampleFeatures:
    tau_values = [safe_float(row.get(f"{leg}_{joint}_tau")) for joint in JOINT_ORDER]
    dq_values = [safe_float(row.get(f"{leg}_{joint}_dq")) for joint in JOINT_ORDER]
    q_values = [safe_float(row.get(f"{leg}_{joint}_q")) for joint in JOINT_ORDER]
    foot_force = safe_float(row.get(f"foot_{leg}"))
    rpy = [safe_float(row.get(axis)) for axis in ("roll", "pitch", "yaw") if axis in row.index]
    gyro = [safe_float(row.get(axis)) for axis in ("gyro_x", "gyro_y", "gyro_z") if axis in row.index]
    acc = [safe_float(row.get(axis)) for axis in ("acc_x", "acc_y", "acc_z") if axis in row.index]

    return SampleFeatures(
        tau_sum=sum(abs(value) for value in tau_values),
        dq_mean=fmean(abs(value) for value in dq_values),
        q_mean=fmean(abs(value) for value in q_values),
        foot_force=foot_force,
        rpy_mean=mean_abs(rpy),
        gyro_mag=vector_magnitude(gyro),
        acc_mag=vector_magnitude(acc),
    )


def load_baseline_samples(baseline_csv: Path) -> dict[str, list[SampleFeatures]]:
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_csv}")

    data = pd.read_csv(baseline_csv)
    if data.empty:
        raise ValueError(f"Baseline CSV contains no rows: {baseline_csv}")

    samples_by_leg: dict[str, list[SampleFeatures]] = {leg: [] for leg in LEG_ORDER}
    for _, row in data.iterrows():
        for leg in LEG_ORDER:
            required_columns = [f"{leg}_{joint}_{metric}" for joint in JOINT_ORDER for metric in ("q", "dq", "tau")]
            if not all(column in data.columns for column in required_columns):
                continue
            if f"foot_{leg}" not in data.columns:
                continue
            samples_by_leg[leg].append(csv_row_features(row, leg))
    return samples_by_leg


def extract_csv_sample(row: pd.Series) -> dict[str, SampleFeatures]:
    samples: dict[str, SampleFeatures] = {}
    for leg in LEG_ORDER:
        if not all(f"{leg}_{joint}_{metric}" in row.index for joint in JOINT_ORDER for metric in ("q", "dq", "tau")):
            continue
        if f"foot_{leg}" not in row.index:
            continue
        samples[leg] = csv_row_features(row, leg)
    return samples


def feature_delta(value: float | None, stats: FeatureStats | None, positive_only: bool = True) -> float:
    if value is None or stats is None:
        return 0.0
    score = (value - stats.mean) / max(stats.std, 1e-6)
    if positive_only:
        return max(0.0, score)
    return score


def feature_delta_below(value: float | None, stats: FeatureStats | None) -> float:
    if value is None or stats is None:
        return 0.0
    score = (stats.mean - value) / max(stats.std, 1e-6)
    return max(0.0, score)


def feature_delta_abs(value: float | None, stats: FeatureStats | None) -> float:
    if value is None or stats is None:
        return 0.0
    score = (value - stats.mean) / max(stats.std, 1e-6)
    return abs(score)


def dominance_delta(value: float, peer_values: list[float], stats: FeatureStats | None) -> float:
    if not peer_values or stats is None:
        return 0.0
    score = (value - max(peer_values)) / max(stats.std, 1e-6)
    return max(0.0, score)


def score_window(
    window_features: dict[str, SampleFeatures],
    baseline_stats: BaselineStats,
) -> dict[str, float]:
    scores: dict[str, float] = {}

    tau_values = {leg: features.tau_sum for leg, features in window_features.items()}
    foot_values = {leg: features.foot_force for leg, features in window_features.items()}

    for leg, features in window_features.items():
        leg_stats = baseline_stats.per_leg.get(leg, {})
        score = 0.0

        score += FEATURE_WEIGHTS["tau_sum"] * feature_delta(features.tau_sum, leg_stats.get("tau_sum"))
        score += FEATURE_WEIGHTS["low_velocity"] * feature_delta_below(features.dq_mean, leg_stats.get("dq_mean"))
        score += FEATURE_WEIGHTS["foot_force"] * feature_delta(features.foot_force, leg_stats.get("foot_force"))
        score += FEATURE_WEIGHTS["q_mean"] * feature_delta_abs(features.q_mean, leg_stats.get("q_mean"))
        score += FEATURE_WEIGHTS["rpy_mean"] * feature_delta_abs(features.rpy_mean, leg_stats.get("rpy_mean"))
        score += FEATURE_WEIGHTS["gyro_mag"] * feature_delta_abs(features.gyro_mag, leg_stats.get("gyro_mag"))
        score += FEATURE_WEIGHTS["acc_dev"] * feature_delta_abs(features.acc_mag, leg_stats.get("acc_mag"))

        others_tau = [value for other_leg, value in tau_values.items() if other_leg != leg]
        if others_tau:
            score += FEATURE_WEIGHTS["tau_dominance"] * dominance_delta(features.tau_sum, others_tau, leg_stats.get("tau_sum"))

        others_foot = [value for other_leg, value in foot_values.items() if other_leg != leg]
        if others_foot:
            score += FEATURE_WEIGHTS["foot_dominance"] * dominance_delta(features.foot_force, others_foot, leg_stats.get("foot_force"))

        scores[leg] = score

    return scores


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_weight = upper - rank
    upper_weight = rank - lower
    return ordered[lower] * lower_weight + ordered[upper] * upper_weight


def calibrate_baseline(baseline_csv: Path, score_percentile: float) -> BaselineStats:
    samples_by_leg = load_baseline_samples(baseline_csv)
    per_leg_stats: dict[str, dict[str, FeatureStats]] = {}

    for leg, samples in samples_by_leg.items():
        per_leg_stats[leg] = leg_feature_stats(samples)

    threshold_scores: dict[str, float] = {leg: 0.0 for leg in LEG_ORDER}
    for leg in LEG_ORDER:
        samples = samples_by_leg.get(leg, [])
        if not samples:
            continue
        scores: list[float] = []
        for sample in samples:
            scores.append(score_window({leg: sample}, BaselineStats(per_leg=per_leg_stats, score_thresholds={}))[leg])
        threshold_scores[leg] = max(percentile(scores, score_percentile), fmean(scores) + 3.0 * (pd.Series(scores).std(ddof=0) if len(scores) > 1 else 0.0))

    return BaselineStats(per_leg=per_leg_stats, score_thresholds=threshold_scores)


class EntanglementEngine:
    def __init__(
        self,
        baseline: BaselineStats,
        window_seconds: float,
        persistence_windows: int,
        dominance_gap: float,
        threshold_scale: float,
    ) -> None:
        self.baseline = baseline
        self.window_seconds = window_seconds
        self.persistence_windows = persistence_windows
        self.dominance_gap = dominance_gap
        self.threshold_scale = threshold_scale
        self.history: dict[str, Deque[tuple[float, SampleFeatures]]] = {leg: deque() for leg in LEG_ORDER}
        self.streaks: dict[str, int] = {leg: 0 for leg in LEG_ORDER}
        self.last_alarm_leg: str | None = None

    def add_sample(self, timestamp: float, features_by_leg: dict[str, SampleFeatures]) -> ScoreResult:
        for leg, features in features_by_leg.items():
            self.history[leg].append((timestamp, features))

        self._prune(timestamp)
        window_features = self._window_average_features()
        scores = score_window(window_features, self.baseline)

        candidate_leg = None
        candidate_score = 0.0
        second_score = 0.0
        ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if ordered_scores:
            candidate_leg, candidate_score = ordered_scores[0]
            if len(ordered_scores) > 1:
                second_score = ordered_scores[1][1]

        alarm_leg: str | None = None
        alarm_score = 0.0
        threshold = 0.0
        streak = 0

        if candidate_leg is not None:
            threshold = self.baseline.score_thresholds.get(candidate_leg, 0.0) * self.threshold_scale
            if candidate_score >= threshold and (candidate_score - second_score) >= self.dominance_gap:
                self.streaks[candidate_leg] += 1
            else:
                self.streaks[candidate_leg] = 0

            streak = self.streaks[candidate_leg]
            if streak >= self.persistence_windows:
                alarm_leg = candidate_leg
                alarm_score = candidate_score

        is_new_alarm = alarm_leg is not None and alarm_leg != self.last_alarm_leg
        self.last_alarm_leg = alarm_leg

        return ScoreResult(
            scores=scores,
            candidate_leg=candidate_leg,
            candidate_score=candidate_score,
            second_score=second_score,
            alarm_leg=alarm_leg,
            alarm_score=alarm_score,
            threshold=threshold,
            streak=streak,
            is_new_alarm=is_new_alarm,
        )

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self.window_seconds
        for leg in LEG_ORDER:
            while self.history[leg] and self.history[leg][0][0] < cutoff:
                self.history[leg].popleft()

    def _window_average_features(self) -> dict[str, SampleFeatures]:
        averaged: dict[str, SampleFeatures] = {}
        for leg, samples in self.history.items():
            if not samples:
                continue
            averages: dict[str, float | None] = {}
            for field_name in SampleFeatures.__annotations__:
                values = [getattr(sample_features, field_name) for _, sample_features in samples if getattr(sample_features, field_name) is not None]
                averages[field_name] = fmean(values) if values else None
            averaged[leg] = SampleFeatures(
                tau_sum=float(averages["tau_sum"] or 0.0),
                dq_mean=float(averages["dq_mean"] or 0.0),
                q_mean=float(averages["q_mean"] or 0.0),
                foot_force=float(averages["foot_force"] or 0.0),
                rpy_mean=averages["rpy_mean"],
                gyro_mag=averages["gyro_mag"],
                acc_mag=averages["acc_mag"],
            )
        return averaged


def print_status(result: ScoreResult, timestamp: float) -> None:
    ordered = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
    top_text = ", ".join(f"{leg}={score:.2f}" for leg, score in ordered)
    print(f"t={timestamp:.3f}s | {top_text}")
    if result.alarm_leg is not None:
        print(
            f"ALARM: likely entanglement on {result.alarm_leg} | "
            f"score={result.alarm_score:.2f} threshold={result.threshold:.2f} streak={result.streak}"
        )


def replay_csv(
    csv_path: Path,
    baseline: BaselineStats,
    window_seconds: float,
    persistence_windows: int,
    dominance_gap: float,
    threshold_scale: float,
    summary_only: bool,
) -> None:
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"Replay CSV contains no rows: {csv_path}")
    if "timestamp" not in data.columns:
        raise ValueError("Replay CSV is missing timestamp")

    timestamps = pd.to_numeric(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Replay CSV has invalid timestamps")
    elapsed = (timestamps - timestamps.iloc[0]).astype(float)

    engine = EntanglementEngine(baseline, window_seconds, persistence_windows, dominance_gap, threshold_scale)
    alarm_count = 0
    alarm_events: list[tuple[float, str, float]] = []
    max_candidate_score = 0.0
    max_candidate_leg: str | None = None
    max_candidate_time = 0.0
    for row_index, (_, row) in enumerate(data.iterrows()):
        features_by_leg = extract_csv_sample(row)
        if not features_by_leg:
            continue
        result = engine.add_sample(float(elapsed.iloc[row_index]), features_by_leg)
        if result.candidate_score > max_candidate_score:
            max_candidate_score = result.candidate_score
            max_candidate_leg = result.candidate_leg
            max_candidate_time = float(elapsed.iloc[row_index])
        if not summary_only and (row_index == 0 or result.is_new_alarm):
            print_status(result, float(elapsed.iloc[row_index]))
        if result.alarm_leg is not None:
            alarm_count += 1
            if result.is_new_alarm:
                alarm_events.append((float(elapsed.iloc[row_index]), result.alarm_leg, result.alarm_score))

    print(f"Finished replay: {csv_path}")
    print(f"Alarms raised: {alarm_count}")
    print(f"Max candidate: {max_candidate_leg} score={max_candidate_score:.2f} at {max_candidate_time:.3f}s")
    if alarm_events:
        first_timestamp, first_leg, first_score = alarm_events[0]
        print(f"First alarm: t={first_timestamp:.3f}s leg={first_leg} score={first_score:.2f}")
    if alarm_events and not summary_only:
        print("Alarm events:")
        for timestamp, leg, score in alarm_events:
            print(f"  t={timestamp:.3f}s leg={leg} score={score:.2f}")


def extract_live_motor_features(msg: Any, foot_order: tuple[str, ...]) -> dict[str, SampleFeatures]:
    if not hasattr(msg, "motor_state"):
        return {}

    motor_state = list(msg.motor_state)
    if len(motor_state) < 12:
        return {}

    foot_values = safe_vector(getattr(msg, "foot_force", None)) or [0.0, 0.0, 0.0, 0.0]
    foot_by_leg = {leg: float(foot_values[index]) if index < len(foot_values) else 0.0 for index, leg in enumerate(foot_order)}

    rpy = get_nested_vector(msg, (("imu_state", "rpy"), ("imu_state", "roll_pitch_yaw"), ("imu", "rpy")))
    gyro = get_nested_vector(msg, (("imu_state", "gyro"), ("imu_state", "gyroscope"), ("imu", "gyro")))
    acc = get_nested_vector(msg, (("imu_state", "accel"), ("imu_state", "accelerometer"), ("imu", "acc")))

    features: dict[str, SampleFeatures] = {}
    for leg, indices in MOTOR_INDEX.items():
        q_values = []
        dq_values = []
        tau_values = []
        for index in indices:
            joint = motor_state[index]
            q_values.append(safe_float(getattr(joint, "q", 0.0)))
            dq_values.append(safe_float(getattr(joint, "dq", 0.0)))
            tau_values.append(safe_float(getattr(joint, "tau_est", getattr(joint, "tau", 0.0))))

        features[leg] = SampleFeatures(
            tau_sum=sum(abs(value) for value in tau_values),
            dq_mean=fmean(abs(value) for value in dq_values),
            q_mean=fmean(abs(value) for value in q_values),
            foot_force=foot_by_leg.get(leg, 0.0),
            rpy_mean=mean_abs(rpy),
            gyro_mag=vector_magnitude(gyro),
            acc_mag=vector_magnitude(acc),
        )

    return features


def run_live_detector(args: argparse.Namespace) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as error:
        raise SystemExit(
            "ROS2 is not available in this environment. Use --replay-csv for offline validation, "
            "or run this script on the robot machine with rclpy installed."
        ) from error

    try:
        from unitree_go.msg import LowState
    except ImportError as error:
        raise SystemExit("unitree_go.msg.LowState is not available in this environment.") from error

    foot_order = tuple(part.strip() for part in args.foot_force_order.split(",") if part.strip())
    if len(foot_order) != 4:
        raise SystemExit("--foot-force-order must contain exactly 4 labels, for example FL,FR,RL,RR")

    baseline = calibrate_baseline(args.baseline_csv, args.score_percentile)
    engine = EntanglementEngine(
        baseline=baseline,
        window_seconds=args.window_seconds,
        persistence_windows=args.persistence_windows,
        dominance_gap=args.dominance_gap,
        threshold_scale=args.threshold_scale,
    )

    print(f"Baseline loaded from {args.baseline_csv}")
    print("Per-leg thresholds:")
    for leg in LEG_ORDER:
        print(f"  {leg}: raw={baseline.score_thresholds.get(leg, 0.0):.2f}, active={baseline.score_thresholds.get(leg, 0.0) * args.threshold_scale:.2f}")
    print("Waiting for /lowstate messages...")

    class LiveDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("entanglement_detector")
            self.subscription = self.create_subscription(LowState, "/lowstate", self.callback, 10)

        def callback(self, msg: LowState) -> None:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            features_by_leg = extract_live_motor_features(msg, foot_order)
            if not features_by_leg:
                self.get_logger().warn("Could not extract enough features from /lowstate")
                return

            result = engine.add_sample(timestamp, features_by_leg)
            ordered = sorted(result.scores.items(), key=lambda item: item[1], reverse=True)
            top_leg, top_score = ordered[0]
            second_score = ordered[1][1] if len(ordered) > 1 else 0.0

            if result.alarm_leg is not None:
                if result.is_new_alarm:
                    self.get_logger().warn(
                        f"ENTANGLEMENT DETECTED leg={result.alarm_leg} score={result.alarm_score:.2f} "
                        f"threshold={result.threshold:.2f} streak={result.streak} "
                        f"scores={{{', '.join(f'{leg}:{score:.2f}' for leg, score in ordered)}}}"
                    )
            elif args.log_normal:
                self.get_logger().info(
                    f"top={top_leg} score={top_score:.2f} second={second_score:.2f} "
                    f"threshold={result.threshold:.2f} streak={result.streak}"
                )

    rclpy.init()
    node = LiveDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = parse_args()
    if args.replay_csv is not None:
        baseline = calibrate_baseline(args.baseline_csv, args.score_percentile)
        print(f"Baseline loaded from {args.baseline_csv}")
        print("Per-leg thresholds:")
        for leg in LEG_ORDER:
            print(f"  {leg}: raw={baseline.score_thresholds.get(leg, 0.0):.2f}, active={baseline.score_thresholds.get(leg, 0.0) * args.threshold_scale:.2f}")
        replay_csv(
            args.replay_csv,
            baseline,
            args.window_seconds,
            args.persistence_windows,
            args.dominance_gap,
            args.threshold_scale,
            args.summary_only,
        )
        return

    run_live_detector(args)


if __name__ == "__main__":
    main()