#!/usr/bin/env python3
"""Live Unitree Go2 entanglement detector focused on thigh torque.

The detector manually calibrates per-leg thigh-torque limits from the first
seconds of a normal walking CSV. In live mode it subscribes to /lowstate and
logs an entanglement event only when the torque change persists inside a short
rolling window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Deque

import pandas as pd


LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_NAMES = {
    "FR": "front_right",
    "FL": "front_left",
    "RR": "back_right",
    "RL": "back_left",
}
JOINT_ORDER = ("hip", "thigh", "calf")

# Unitree Go2 lowstate motor order used by the sample code:
# FR: 0,1,2 | FL: 3,4,5 | RR: 6,7,8 | RL: 9,10,11
MOTOR_INDEX = {
    "FR": {"hip": 0, "thigh": 1, "calf": 2},
    "FL": {"hip": 3, "thigh": 4, "calf": 5},
    "RR": {"hip": 6, "thigh": 7, "calf": 8},
    "RL": {"hip": 9, "thigh": 10, "calf": 11},
}

DEFAULT_WALKING_CSV = Path("data/go2_lowstate_walking_v2.csv")
DEFAULT_ENTANGLEMENT_CASES = (
    (Path("data/go2_lowstate_back_left_leg.csv"), ("RL",)),
    (Path("data/go2_lowstate_back_right_leg.csv"), ("RR",)),
    (Path("data/go2_lowstate_back_both_leg.csv"), ("RR", "RL")),
    (Path("data/go2_lowstate_front_both_leg.csv"), ("FR", "FL")),
    (Path("data/go2_lowstate_front_left.csv"), ("FL",)),
    (Path("data/go2_lowstate_front_right.csv"), ("FR",)),
)
DEFAULT_LOG_PATH = Path("entanglement_events.csv")
DEFAULT_MODEL_CONFIG_PATH = Path("data/thigh_torque_detector_params.json")
DEFAULT_CALIBRATION_SECONDS = 1e9
DEFAULT_WINDOW_SECONDS = 0.20
DEFAULT_PERSISTENCE_SECONDS = 0.12
DEFAULT_REQUIRED_WINDOW_FRACTION = 0.75
DEFAULT_COOLDOWN_SECONDS = 1.0
DEFAULT_WALKING_LOWER_PERCENTILE = 0.5
DEFAULT_ENTANGLEMENT_LOWER_PERCENTILE = 5.0
DEFAULT_THRESHOLD_BLEND = 0.5
DEFAULT_RELATIVE_BASELINE_SECONDS = 2.0

# CSV foot-force columns are named by leg. Unitree message ordering can vary by
# SDK/bridge, so keep it configurable; this order matches the existing scripts.
DEFAULT_FOOT_FORCE_ORDER = ("FL", "FR", "RL", "RR")
OPPOSITE_LEG = {
    "FR": "FL",
    "FL": "FR",
    "RR": "RL",
    "RL": "RR",
}


@dataclass(frozen=True)
class TorqueSample:
    timestamp: float
    thigh_tau: float
    thigh_dq: float
    torque_abs_sum: float
    dq_abs_mean: float
    foot_force: float


@dataclass(frozen=True)
class WindowFeatures:
    abs_mean: float
    abs_max: float
    rms: float
    peak_to_peak: float
    slope_abs_mean: float
    low_dq: float
    foot_mean: float


@dataclass(frozen=True)
class FeatureStats:
    median: float
    scale: float


@dataclass(frozen=True)
class LegModel:
    feature_stats: dict[str, FeatureStats]
    threshold: float
    torque_threshold: float
    polarity: str
    calibration_median_tau: float
    walking_lower_tau: float
    walking_upper_tau: float
    entanglement_lower_tau: float | None
    calibration_min_tau: float
    calibration_max_tau: float
    torque_scale: float
    effort_threshold: float
    effort_scale: float
    walking_upper_effort: float
    entanglement_upper_effort: float | None
    stalled_effort_threshold: float
    stalled_effort_scale: float
    walking_upper_stalled_effort: float
    entanglement_upper_stalled_effort: float | None
    relative_tau_threshold: float
    relative_tau_scale: float
    walking_upper_relative_tau: float
    entanglement_upper_relative_tau: float | None


@dataclass(frozen=True)
class DetectionResult:
    scores: dict[str, float]
    candidate_leg: str | None
    candidate_score: float
    threshold: float
    alarm_leg: str | None
    streak: int
    is_new_alarm: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Go2 entanglement detector using thigh-torque time-series anomalies.")
    parser.add_argument("--walking-csv", type=Path, default=DEFAULT_WALKING_CSV, help="Normal walking CSV used as the baseline.")
    parser.add_argument(
        "--entanglement-case",
        action="append",
        default=None,
        metavar="CSV:LEG",
        help="Labeled entanglement CSV used to place signed thigh-torque thresholds, for example data/log.csv:RR or data/log.csv:FR,FL. Can be repeated.",
    )
    parser.add_argument("--replay-csv", type=Path, default=None, help="Replay a CSV instead of subscribing to /lowstate.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH, help="Fixed detector parameter JSON. Loaded when it exists unless --force-calibrate is used.")
    parser.add_argument("--force-calibrate", action="store_true", help="Ignore --model-config and rebuild thresholds from labeled CSVs.")
    parser.add_argument("--no-save-model-config", dest="save_model_config", action="store_false", help="Do not write calibrated thresholds to --model-config.")
    parser.set_defaults(save_model_config=True)
    parser.add_argument("--calibration-seconds", type=float, default=DEFAULT_CALIBRATION_SECONDS, help="Use walking data from t=0 up to this time for manual calibration.")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="Sliding time-series window size.")
    parser.add_argument("--persistence-seconds", type=float, default=DEFAULT_PERSISTENCE_SECONDS, help="Continuous signed-threshold duration required inside the rolling window.")
    parser.add_argument("--required-window-fraction", type=float, default=DEFAULT_REQUIRED_WINDOW_FRACTION, help="Fraction of samples in the rolling window that must exceed the signed thigh-torque threshold.")
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS, help="Minimum time between logged events for the same leg.")
    parser.add_argument("--walking-lower-percentile", type=float, default=DEFAULT_WALKING_LOWER_PERCENTILE, help="Lower percentile of walking thigh_tau used as the normal lower guard.")
    parser.add_argument("--entanglement-lower-percentile", type=float, default=DEFAULT_ENTANGLEMENT_LOWER_PERCENTILE, help="Tail percentile of labeled entanglement thigh_tau used as the spike reference.")
    parser.add_argument("--threshold-blend", type=float, default=DEFAULT_THRESHOLD_BLEND, help="0 uses entanglement spike value, 1 uses walking guard; 0.5 is the midpoint.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="CSV file where live/replay entanglement events are appended.")
    parser.add_argument("--log-normal", action="store_true", help="Print normal status lines as well as alarms.")
    parser.add_argument(
        "--foot-force-order",
        default=",".join(DEFAULT_FOOT_FORCE_ORDER),
        help="Order of msg.foot_force values, comma-separated, for example FL,FR,RL,RR.",
    )
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = max(0.0, min(100.0, q)) / 100.0 * (len(clean) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    return clean[lower] * (upper - rank) + clean[upper] * (rank - lower)


def median(values: list[float]) -> float:
    return percentile(values, 50.0)


def robust_stats(values: list[float]) -> FeatureStats:
    if not values:
        return FeatureStats(median=0.0, scale=1.0)
    center = median(values)
    deviations = [abs(value - center) for value in values]
    mad_sigma = 1.4826 * median(deviations)
    if mad_sigma > 1e-6:
        scale = mad_sigma
    elif len(values) > 1:
        mean_value = fmean(values)
        variance = fmean((value - mean_value) ** 2 for value in values)
        scale = math.sqrt(variance) if variance > 1e-12 else 1.0
    else:
        scale = 1.0
    return FeatureStats(median=center, scale=max(scale, 1e-6))


def z_above(value: float, stats: FeatureStats) -> float:
    return max(0.0, (value - stats.median) / stats.scale)


def z_below(value: float, stats: FeatureStats) -> float:
    return max(0.0, (stats.median - value) / stats.scale)


def build_window_features(samples: list[TorqueSample]) -> WindowFeatures | None:
    if len(samples) < 2:
        return None

    tau_values = [sample.thigh_tau for sample in samples]
    abs_tau = [abs(value) for value in tau_values]
    dq_values = [abs(sample.thigh_dq) for sample in samples]
    foot_values = [sample.foot_force for sample in samples]

    slopes: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        dt = current.timestamp - previous.timestamp
        if dt > 1e-6:
            slopes.append(abs(current.thigh_tau - previous.thigh_tau) / dt)

    return WindowFeatures(
        abs_mean=fmean(abs_tau),
        abs_max=max(abs_tau),
        rms=math.sqrt(fmean(value * value for value in tau_values)),
        peak_to_peak=max(tau_values) - min(tau_values),
        slope_abs_mean=fmean(slopes) if slopes else 0.0,
        low_dq=max(0.0, 1.0 / (fmean(dq_values) + 0.05)),
        foot_mean=fmean(foot_values),
    )


def feature_dict(features: WindowFeatures) -> dict[str, float]:
    return {
        "abs_mean": features.abs_mean,
        "abs_max": features.abs_max,
        "rms": features.rms,
        "peak_to_peak": features.peak_to_peak,
        "slope_abs_mean": features.slope_abs_mean,
        "low_dq": features.low_dq,
        "foot_mean": features.foot_mean,
    }


def score_features(features: WindowFeatures, model: LegModel) -> float:
    values = feature_dict(features)
    score = 0.0

    # Thigh torque is intentionally dominant: sustained magnitude, peaks, RMS,
    # waveform spread, and sharp changes are the main anomaly cues.
    score += 3.2 * z_above(values["abs_mean"], model.feature_stats["abs_mean"])
    score += 2.4 * z_above(values["abs_max"], model.feature_stats["abs_max"])
    score += 2.0 * z_above(values["rms"], model.feature_stats["rms"])
    score += 1.4 * z_above(values["peak_to_peak"], model.feature_stats["peak_to_peak"])
    score += 1.2 * z_above(values["slope_abs_mean"], model.feature_stats["slope_abs_mean"])

    # These are secondary stabilizers: entanglement commonly causes high effort
    # with slower motion/contact load, but they should not override thigh torque.
    score += 0.6 * z_above(values["low_dq"], model.feature_stats["low_dq"])
    score += 0.4 * z_above(values["foot_mean"], model.feature_stats["foot_mean"])
    return score


def read_csv_samples(csv_path: Path) -> tuple[dict[str, list[TorqueSample]], float]:
    data = pd.read_csv(csv_path)
    if data.empty:
        raise ValueError(f"CSV contains no rows: {csv_path}")
    if "timestamp" not in data.columns:
        raise ValueError(f"CSV is missing timestamp: {csv_path}")

    timestamps = pd.to_numeric(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"CSV contains invalid timestamps: {csv_path}")
    elapsed = (timestamps - timestamps.iloc[0]).astype(float)

    samples_by_leg: dict[str, list[TorqueSample]] = {leg: [] for leg in LEG_ORDER}
    for row_index, row in data.iterrows():
        timestamp = float(elapsed.iloc[row_index])
        for leg in LEG_ORDER:
            tau_col = f"{leg}_thigh_tau"
            dq_col = f"{leg}_thigh_dq"
            foot_col = f"foot_{leg}"
            if tau_col not in data.columns or dq_col not in data.columns:
                continue
            tau_cols = [f"{leg}_{joint}_tau" for joint in JOINT_ORDER if f"{leg}_{joint}_tau" in data.columns]
            dq_cols = [f"{leg}_{joint}_dq" for joint in JOINT_ORDER if f"{leg}_{joint}_dq" in data.columns]
            torque_abs_sum = sum(abs(safe_float(row.get(column))) for column in tau_cols) if tau_cols else abs(safe_float(row.get(tau_col)))
            dq_abs_mean = fmean(abs(safe_float(row.get(column))) for column in dq_cols) if dq_cols else abs(safe_float(row.get(dq_col)))
            samples_by_leg[leg].append(
                TorqueSample(
                    timestamp=timestamp,
                    thigh_tau=safe_float(row.get(tau_col)),
                    thigh_dq=safe_float(row.get(dq_col)),
                    torque_abs_sum=torque_abs_sum,
                    dq_abs_mean=dq_abs_mean,
                    foot_force=safe_float(row.get(foot_col)),
                )
            )
    return samples_by_leg, float(elapsed.iloc[-1]) if len(elapsed) else 0.0


def rolling_windows(samples: list[TorqueSample], window_seconds: float) -> list[WindowFeatures]:
    history: Deque[TorqueSample] = deque()
    windows: list[WindowFeatures] = []
    for sample in samples:
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        features = build_window_features(list(history))
        if features is not None:
            windows.append(features)
    return windows


def relative_tau_changes(samples: list[TorqueSample], polarity: str, baseline_seconds: float = DEFAULT_RELATIVE_BASELINE_SECONDS) -> list[float]:
    history: Deque[TorqueSample] = deque()
    changes: list[float] = []
    for sample in samples:
        history.append(sample)
        cutoff = sample.timestamp - baseline_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if polarity == "down":
            reference = max(history_sample.thigh_tau for history_sample in history)
            changes.append(max(0.0, reference - sample.thigh_tau))
        else:
            reference = min(history_sample.thigh_tau for history_sample in history)
            changes.append(max(0.0, sample.thigh_tau - reference))
    return changes


def parse_entanglement_cases(raw_cases: list[str] | None) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    if not raw_cases:
        return DEFAULT_ENTANGLEMENT_CASES

    parsed: list[tuple[Path, tuple[str, ...]]] = []
    for raw_case in raw_cases:
        if ":" not in raw_case:
            raise SystemExit(f"Invalid --entanglement-case {raw_case!r}; expected CSV:LEG or CSV:LEG,LEG")
        csv_text, legs_text = raw_case.rsplit(":", 1)
        legs = tuple(part.strip().upper() for part in legs_text.split(",") if part.strip())
        if not legs or any(leg not in LEG_ORDER for leg in legs):
            raise SystemExit(f"Invalid leg list {legs_text!r}; expected labels from {', '.join(LEG_ORDER)}")
        parsed.append((Path(csv_text), legs))
    return tuple(parsed)


def default_polarity(leg: str) -> str:
    return "up" if leg in ("FR", "FL") else "down"


def calibrate_models(
    walking_csv: Path,
    entanglement_cases: tuple[tuple[Path, tuple[str, ...]], ...],
    calibration_seconds: float,
    walking_lower_percentile: float,
    entanglement_lower_percentile: float,
    threshold_blend: float,
    polarity_by_leg: dict[str, str] | None = None,
) -> dict[str, LegModel]:
    walking_samples, _ = read_csv_samples(walking_csv)
    entanglement_tau_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    entanglement_effort_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    entanglement_stalled_effort_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    entanglement_relative_down_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    entanglement_relative_up_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    non_entanglement_tau_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    non_entanglement_effort_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    non_entanglement_stalled_effort_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    non_entanglement_relative_down_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    non_entanglement_relative_up_by_leg: dict[str, list[float]] = {leg: [] for leg in LEG_ORDER}
    for csv_path, label_legs in entanglement_cases:
        if not csv_path.exists():
            continue
        case_samples, _ = read_csv_samples(csv_path)
        label_set = set(label_legs)
        for label_leg in label_legs:
            label_samples = case_samples.get(label_leg, [])
            entanglement_tau_by_leg[label_leg].extend(sample.thigh_tau for sample in label_samples)
            entanglement_effort_by_leg[label_leg].extend(sample.torque_abs_sum for sample in label_samples)
            entanglement_stalled_effort_by_leg[label_leg].extend(sample.torque_abs_sum / (sample.dq_abs_mean + 0.05) for sample in label_samples)
            entanglement_relative_down_by_leg[label_leg].extend(relative_tau_changes(label_samples, "down"))
            entanglement_relative_up_by_leg[label_leg].extend(relative_tau_changes(label_samples, "up"))
        for label_leg in label_legs:
            opposite_leg = OPPOSITE_LEG[label_leg]
            if opposite_leg not in label_set:
                opposite_samples = case_samples.get(opposite_leg, [])
                non_entanglement_tau_by_leg[opposite_leg].extend(sample.thigh_tau for sample in opposite_samples)
                non_entanglement_effort_by_leg[opposite_leg].extend(sample.torque_abs_sum for sample in opposite_samples)
                non_entanglement_stalled_effort_by_leg[opposite_leg].extend(sample.torque_abs_sum / (sample.dq_abs_mean + 0.05) for sample in opposite_samples)
                non_entanglement_relative_down_by_leg[opposite_leg].extend(relative_tau_changes(opposite_samples, "down"))
                non_entanglement_relative_up_by_leg[opposite_leg].extend(relative_tau_changes(opposite_samples, "up"))

    models: dict[str, LegModel] = {}
    blend = max(0.0, min(1.0, threshold_blend))
    for leg in LEG_ORDER:
        calibration_samples = [sample for sample in walking_samples[leg] if sample.timestamp <= calibration_seconds]
        if not calibration_samples:
            raise ValueError(f"No usable walking windows for {leg}; check {walking_csv}")
        walking_tau = [sample.thigh_tau for sample in calibration_samples]
        walking_effort = [sample.torque_abs_sum for sample in calibration_samples]
        walking_stalled_effort = [sample.torque_abs_sum / (sample.dq_abs_mean + 0.05) for sample in calibration_samples]
        polarity = (polarity_by_leg or {}).get(leg, default_polarity(leg))
        if polarity not in ("down", "up"):
            raise ValueError(f"Invalid polarity for {leg}: {polarity!r}")
        walking_relative_tau = relative_tau_changes(calibration_samples, polarity)
        walking_lower = percentile(walking_tau, walking_lower_percentile)
        walking_upper = percentile(walking_tau, 100.0 - walking_lower_percentile)
        negative_tau = walking_tau + non_entanglement_tau_by_leg[leg]
        negative_lower = percentile(negative_tau, walking_lower_percentile)
        negative_upper = percentile(negative_tau, 100.0 - walking_lower_percentile)
        negative_effort = walking_effort + non_entanglement_effort_by_leg[leg]
        walking_upper_effort = percentile(walking_effort, 100.0 - walking_lower_percentile)
        negative_upper_effort = percentile(negative_effort, 100.0 - walking_lower_percentile)
        negative_stalled_effort = walking_stalled_effort + non_entanglement_stalled_effort_by_leg[leg]
        walking_upper_stalled_effort = percentile(walking_stalled_effort, 100.0 - walking_lower_percentile)
        negative_upper_stalled_effort = percentile(negative_stalled_effort, 100.0 - walking_lower_percentile)
        non_entanglement_relative_tau = non_entanglement_relative_down_by_leg[leg] if polarity == "down" else non_entanglement_relative_up_by_leg[leg]
        negative_relative_tau = walking_relative_tau + non_entanglement_relative_tau
        walking_upper_relative_tau = percentile(walking_relative_tau, 100.0 - walking_lower_percentile)
        negative_upper_relative_tau = percentile(negative_relative_tau, 100.0 - walking_lower_percentile)
        entanglement_lower = None
        if entanglement_tau_by_leg[leg]:
            entanglement_lower = percentile(
                entanglement_tau_by_leg[leg],
                entanglement_lower_percentile if polarity == "down" else 100.0 - entanglement_lower_percentile,
            )
        entanglement_upper_effort = None
        if entanglement_effort_by_leg[leg]:
            entanglement_upper_effort = percentile(entanglement_effort_by_leg[leg], 100.0 - entanglement_lower_percentile)
        entanglement_upper_stalled_effort = None
        if entanglement_stalled_effort_by_leg[leg]:
            entanglement_upper_stalled_effort = percentile(entanglement_stalled_effort_by_leg[leg], 100.0 - entanglement_lower_percentile)
        entanglement_relative_tau = entanglement_relative_down_by_leg[leg] if polarity == "down" else entanglement_relative_up_by_leg[leg]
        entanglement_upper_relative_tau = None
        if entanglement_relative_tau:
            entanglement_upper_relative_tau = percentile(entanglement_relative_tau, 100.0 - entanglement_lower_percentile)

        walking_stats = robust_stats(walking_tau)
        effort_stats = robust_stats(walking_effort)
        stalled_effort_stats = robust_stats(walking_stalled_effort)
        relative_tau_stats = robust_stats(walking_relative_tau)
        negative_guard = min(walking_lower, negative_lower) if polarity == "down" else max(walking_upper, negative_upper)
        if polarity == "down" and entanglement_lower is not None and entanglement_lower < negative_guard:
            torque_threshold = entanglement_lower * (1.0 - blend) + negative_guard * blend
        elif polarity == "up" and entanglement_lower is not None and entanglement_lower > negative_guard:
            torque_threshold = entanglement_lower * (1.0 - blend) + negative_guard * blend
        else:
            torque_threshold = negative_guard - 2.5 * walking_stats.scale if polarity == "down" else negative_guard + 2.5 * walking_stats.scale
        effort_guard = max(walking_upper_effort, negative_upper_effort)
        if entanglement_upper_effort is not None and entanglement_upper_effort > effort_guard:
            effort_threshold = entanglement_upper_effort * (1.0 - blend) + effort_guard * blend
        else:
            effort_threshold = effort_guard + 2.5 * effort_stats.scale
        stalled_effort_guard = max(walking_upper_stalled_effort, negative_upper_stalled_effort)
        if entanglement_upper_stalled_effort is not None and entanglement_upper_stalled_effort > stalled_effort_guard:
            stalled_effort_threshold = entanglement_upper_stalled_effort * (1.0 - blend) + stalled_effort_guard * blend
        else:
            stalled_effort_threshold = stalled_effort_guard + 2.5 * stalled_effort_stats.scale
        relative_tau_guard = max(walking_upper_relative_tau, negative_upper_relative_tau)
        if entanglement_upper_relative_tau is not None and entanglement_upper_relative_tau > relative_tau_guard:
            relative_tau_threshold = entanglement_upper_relative_tau * (1.0 - blend) + relative_tau_guard * blend
        else:
            relative_tau_threshold = relative_tau_guard + 2.5 * relative_tau_stats.scale

        dummy_stats = {
            "abs_mean": walking_stats,
            "abs_max": walking_stats,
            "rms": walking_stats,
            "peak_to_peak": FeatureStats(median=0.0, scale=1.0),
            "slope_abs_mean": FeatureStats(median=0.0, scale=1.0),
            "low_dq": FeatureStats(median=0.0, scale=1.0),
            "foot_mean": FeatureStats(median=0.0, scale=1.0),
        }
        walking_guard = negative_guard
        torque_scale = max(abs(walking_guard - torque_threshold), walking_stats.scale, 1e-6)
        effort_scale = max(abs(effort_threshold - effort_guard), effort_stats.scale, 1e-6)
        stalled_effort_scale = max(abs(stalled_effort_threshold - stalled_effort_guard), stalled_effort_stats.scale, 1e-6)
        relative_tau_scale = max(abs(relative_tau_threshold - relative_tau_guard), relative_tau_stats.scale, 1e-6)
        models[leg] = LegModel(
            feature_stats=dummy_stats,
            threshold=torque_threshold,
            torque_threshold=torque_threshold,
            polarity=polarity,
            calibration_median_tau=median(walking_tau),
            walking_lower_tau=walking_lower,
            walking_upper_tau=walking_upper,
            entanglement_lower_tau=entanglement_lower,
            calibration_min_tau=min(walking_tau),
            calibration_max_tau=max(walking_tau),
            torque_scale=torque_scale,
            effort_threshold=effort_threshold,
            effort_scale=effort_scale,
            walking_upper_effort=walking_upper_effort,
            entanglement_upper_effort=entanglement_upper_effort,
            stalled_effort_threshold=stalled_effort_threshold,
            stalled_effort_scale=stalled_effort_scale,
            walking_upper_stalled_effort=walking_upper_stalled_effort,
            entanglement_upper_stalled_effort=entanglement_upper_stalled_effort,
            relative_tau_threshold=relative_tau_threshold,
            relative_tau_scale=relative_tau_scale,
            walking_upper_relative_tau=walking_upper_relative_tau,
            entanglement_upper_relative_tau=entanglement_upper_relative_tau,
        )

    return models


def model_to_json(model: LegModel) -> dict[str, Any]:
    return {
        "threshold": model.torque_threshold,
        "torque_threshold": model.torque_threshold,
        "polarity": model.polarity,
        "calibration_median_tau": model.calibration_median_tau,
        "walking_lower_tau": model.walking_lower_tau,
        "walking_upper_tau": model.walking_upper_tau,
        "entanglement_extreme_tau": model.entanglement_lower_tau,
        "calibration_min_tau": model.calibration_min_tau,
        "calibration_max_tau": model.calibration_max_tau,
        "torque_scale": model.torque_scale,
        "effort_threshold": model.effort_threshold,
        "effort_scale": model.effort_scale,
        "walking_upper_effort": model.walking_upper_effort,
        "entanglement_upper_effort": model.entanglement_upper_effort,
        "stalled_effort_threshold": model.stalled_effort_threshold,
        "stalled_effort_scale": model.stalled_effort_scale,
        "walking_upper_stalled_effort": model.walking_upper_stalled_effort,
        "entanglement_upper_stalled_effort": model.entanglement_upper_stalled_effort,
        "relative_tau_threshold": model.relative_tau_threshold,
        "relative_tau_scale": model.relative_tau_scale,
        "walking_upper_relative_tau": model.walking_upper_relative_tau,
        "entanglement_upper_relative_tau": model.entanglement_upper_relative_tau,
    }


def save_model_config(config_path: Path, models: dict[str, LegModel], args: argparse.Namespace) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "walking_csv": str(args.walking_csv),
        "calibration_seconds": args.calibration_seconds,
        "window_seconds": args.window_seconds,
        "persistence_seconds": args.persistence_seconds,
        "required_window_fraction": args.required_window_fraction,
        "cooldown_seconds": args.cooldown_seconds,
        "walking_lower_percentile": args.walking_lower_percentile,
        "entanglement_lower_percentile": args.entanglement_lower_percentile,
        "threshold_blend": args.threshold_blend,
        "models": {leg: model_to_json(model) for leg, model in models.items()},
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_model_config(config_path: Path) -> tuple[dict[str, LegModel], dict[str, Any]]:
    payload = json.loads(config_path.read_text())
    raw_models = payload.get("models", {})
    models: dict[str, LegModel] = {}
    dummy_stats = {
        "abs_mean": FeatureStats(median=0.0, scale=1.0),
        "abs_max": FeatureStats(median=0.0, scale=1.0),
        "rms": FeatureStats(median=0.0, scale=1.0),
        "peak_to_peak": FeatureStats(median=0.0, scale=1.0),
        "slope_abs_mean": FeatureStats(median=0.0, scale=1.0),
        "low_dq": FeatureStats(median=0.0, scale=1.0),
        "foot_mean": FeatureStats(median=0.0, scale=1.0),
    }
    for leg in LEG_ORDER:
        raw = raw_models.get(leg)
        if raw is None:
            raise ValueError(f"Model config is missing leg {leg}: {config_path}")
        threshold = safe_float(raw.get("torque_threshold", raw.get("threshold")))
        polarity = str(raw.get("polarity", default_polarity(leg))).lower()
        if polarity not in ("down", "up"):
            raise ValueError(f"Invalid polarity for {leg}: {polarity!r}")
        models[leg] = LegModel(
            feature_stats=dummy_stats,
            threshold=threshold,
            torque_threshold=threshold,
            polarity=polarity,
            calibration_median_tau=safe_float(raw.get("calibration_median_tau")),
            walking_lower_tau=safe_float(raw.get("walking_lower_tau")),
            walking_upper_tau=safe_float(raw.get("walking_upper_tau")),
            entanglement_lower_tau=raw.get("entanglement_extreme_tau"),
            calibration_min_tau=safe_float(raw.get("calibration_min_tau")),
            calibration_max_tau=safe_float(raw.get("calibration_max_tau")),
            torque_scale=max(safe_float(raw.get("torque_scale"), 1.0), 1e-6),
            effort_threshold=safe_float(raw.get("effort_threshold"), 1e18),
            effort_scale=max(safe_float(raw.get("effort_scale"), 1.0), 1e-6),
            walking_upper_effort=safe_float(raw.get("walking_upper_effort")),
            entanglement_upper_effort=raw.get("entanglement_upper_effort"),
            stalled_effort_threshold=safe_float(raw.get("stalled_effort_threshold"), 1e18),
            stalled_effort_scale=max(safe_float(raw.get("stalled_effort_scale"), 1.0), 1e-6),
            walking_upper_stalled_effort=safe_float(raw.get("walking_upper_stalled_effort")),
            entanglement_upper_stalled_effort=raw.get("entanglement_upper_stalled_effort"),
            relative_tau_threshold=safe_float(raw.get("relative_tau_threshold"), 1e18),
            relative_tau_scale=max(safe_float(raw.get("relative_tau_scale"), 1.0), 1e-6),
            walking_upper_relative_tau=safe_float(raw.get("walking_upper_relative_tau")),
            entanglement_upper_relative_tau=raw.get("entanglement_upper_relative_tau"),
        )
    return models, payload


class ThighTorqueDetector:
    def __init__(
        self,
        models: dict[str, LegModel],
        window_seconds: float,
        persistence_seconds: float,
        required_window_fraction: float,
    ) -> None:
        self.models = models
        self.window_seconds = window_seconds
        self.persistence_seconds = persistence_seconds
        self.required_window_fraction = required_window_fraction
        self.history: dict[str, Deque[TorqueSample]] = {leg: deque() for leg in LEG_ORDER}
        self.baseline_history: dict[str, Deque[TorqueSample]] = {leg: deque() for leg in LEG_ORDER}
        self.streaks: dict[str, int] = {leg: 0 for leg in LEG_ORDER}
        self.last_alarm_leg: str | None = None

    def add_samples(self, samples_by_leg: dict[str, TorqueSample]) -> DetectionResult:
        timestamp = max((sample.timestamp for sample in samples_by_leg.values()), default=0.0)
        for leg, sample in samples_by_leg.items():
            self.history[leg].append(sample)
            self.baseline_history[leg].append(sample)

        cutoff = timestamp - self.window_seconds
        baseline_cutoff = timestamp - max(DEFAULT_RELATIVE_BASELINE_SECONDS, self.window_seconds)
        for leg in LEG_ORDER:
            while self.history[leg] and self.history[leg][0].timestamp < cutoff:
                self.history[leg].popleft()
            while self.baseline_history[leg] and self.baseline_history[leg][0].timestamp < baseline_cutoff:
                self.baseline_history[leg].popleft()

        scores: dict[str, float] = {}
        persistent_seconds: dict[str, float] = {}
        window_fractions: dict[str, float] = {}
        for leg in LEG_ORDER:
            samples = list(self.history[leg])
            baseline_samples = list(self.baseline_history[leg])
            model = self.models[leg]
            threshold = model.torque_threshold
            exceed_samples = [sample for sample in samples if self._is_any_exceedance(sample, model, baseline_samples)]
            window_fractions[leg] = len(exceed_samples) / len(samples) if samples else 0.0
            persistent_seconds[leg] = self._latest_continuous_exceedance_seconds(samples, model, baseline_samples)
            if not samples:
                scores[leg] = 0.0
            else:
                if model.polarity == "down":
                    extreme_tau = min(sample.thigh_tau for sample in samples)
                    level_margin = threshold - extreme_tau
                else:
                    extreme_tau = max(sample.thigh_tau for sample in samples)
                    level_margin = extreme_tau - threshold
                thigh_score_from_level = 1.0 + max(0.0, level_margin) / model.torque_scale
                extreme_effort = max(sample.torque_abs_sum for sample in samples)
                effort_margin = extreme_effort - model.effort_threshold
                effort_score_from_level = 1.0 + max(0.0, effort_margin) / model.effort_scale
                extreme_stalled_effort = max(sample.torque_abs_sum / (sample.dq_abs_mean + 0.05) for sample in samples)
                stalled_effort_margin = extreme_stalled_effort - model.stalled_effort_threshold
                stalled_effort_score_from_level = 1.0 + max(0.0, stalled_effort_margin) / model.stalled_effort_scale
                relative_margin = self._relative_tau_margin(samples, baseline_samples, model)
                relative_tau_score_from_level = 1.0 + max(0.0, relative_margin) / model.relative_tau_scale
                score_from_level = max(thigh_score_from_level, effort_score_from_level, stalled_effort_score_from_level, relative_tau_score_from_level)
                score_from_fraction = window_fractions[leg] / max(self.required_window_fraction, 1e-6)
                score_from_persistence = persistent_seconds[leg] / max(self.persistence_seconds, 1e-6)
                scores[leg] = min(score_from_level, score_from_fraction, score_from_persistence)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate_leg = ordered[0][0] if ordered else None
        candidate_score = ordered[0][1] if ordered else 0.0
        threshold = 1.0 if candidate_leg else 0.0
        alarm_leg: str | None = None
        streak = 0

        if candidate_leg is not None and candidate_score >= threshold:
            self.streaks[candidate_leg] += 1
        elif candidate_leg is not None:
            self.streaks[candidate_leg] = 0

        if candidate_leg is not None:
            streak = self.streaks[candidate_leg]
            if streak >= 1:
                alarm_leg = candidate_leg

        is_new_alarm = alarm_leg is not None and alarm_leg != self.last_alarm_leg
        self.last_alarm_leg = alarm_leg
        return DetectionResult(scores=scores, candidate_leg=candidate_leg, candidate_score=candidate_score, threshold=threshold, alarm_leg=alarm_leg, streak=streak, is_new_alarm=is_new_alarm)

    @staticmethod
    def _is_exceedance(thigh_tau: float, model: LegModel) -> bool:
        if model.polarity == "down":
            return thigh_tau <= model.torque_threshold
        return thigh_tau >= model.torque_threshold

    @staticmethod
    def _is_effort_exceedance(sample: TorqueSample, model: LegModel) -> bool:
        return sample.torque_abs_sum >= model.effort_threshold

    @staticmethod
    def _is_stalled_effort_exceedance(sample: TorqueSample, model: LegModel) -> bool:
        return sample.torque_abs_sum / (sample.dq_abs_mean + 0.05) >= model.stalled_effort_threshold

    def _is_relative_tau_exceedance(self, sample: TorqueSample, baseline_samples: list[TorqueSample], model: LegModel) -> bool:
        if not baseline_samples:
            return False
        if model.polarity == "down":
            reference = max(baseline_sample.thigh_tau for baseline_sample in baseline_samples)
            return reference - sample.thigh_tau >= model.relative_tau_threshold
        reference = min(baseline_sample.thigh_tau for baseline_sample in baseline_samples)
        return sample.thigh_tau - reference >= model.relative_tau_threshold

    def _relative_tau_margin(self, samples: list[TorqueSample], baseline_samples: list[TorqueSample], model: LegModel) -> float:
        if not samples or not baseline_samples:
            return -model.relative_tau_threshold
        if model.polarity == "down":
            reference = max(baseline_sample.thigh_tau for baseline_sample in baseline_samples)
            extreme = min(sample.thigh_tau for sample in samples)
            return reference - extreme - model.relative_tau_threshold
        reference = min(baseline_sample.thigh_tau for baseline_sample in baseline_samples)
        extreme = max(sample.thigh_tau for sample in samples)
        return extreme - reference - model.relative_tau_threshold

    def _is_any_exceedance(self, sample: TorqueSample, model: LegModel, baseline_samples: list[TorqueSample]) -> bool:
        return (
            self._is_exceedance(sample.thigh_tau, model)
            or self._is_effort_exceedance(sample, model)
            or self._is_stalled_effort_exceedance(sample, model)
            or self._is_relative_tau_exceedance(sample, baseline_samples, model)
        )

    def _latest_continuous_exceedance_seconds(self, samples: list[TorqueSample], model: LegModel, baseline_samples: list[TorqueSample]) -> float:
        if len(samples) < 2:
            return 0.0
        latest = samples[-1]
        if not self._is_any_exceedance(latest, model, baseline_samples):
            return 0.0
        start = latest.timestamp
        for sample in reversed(samples[:-1]):
            if not self._is_any_exceedance(sample, model, baseline_samples):
                break
            start = sample.timestamp
        return max(0.0, latest.timestamp - start)


def init_event_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() and log_path.stat().st_size > 0:
        return
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["wall_time", "elapsed_or_ros_time", "leg", "leg_name", "score", "threshold", "streak", "scores"])


def append_event(log_path: Path, timestamp: float, result: DetectionResult) -> None:
    if result.alarm_leg is None:
        return
    with log_path.open("a", newline="") as log_file:
        writer = csv.writer(log_file)
        score_text = ";".join(f"{leg}:{score:.3f}" for leg, score in sorted(result.scores.items()))
        writer.writerow([
            f"{time.time():.3f}",
            f"{timestamp:.3f}",
            result.alarm_leg,
            LEG_NAMES[result.alarm_leg],
            f"{result.candidate_score:.3f}",
            f"{result.threshold:.3f}",
            result.streak,
            score_text,
        ])


def replay_csv(csv_path: Path, detector: ThighTorqueDetector, log_path: Path, log_normal: bool, cooldown_seconds: float) -> None:
    samples_by_leg, _ = read_csv_samples(csv_path)
    events = 0
    max_score = 0.0
    max_leg: str | None = None
    last_logged: dict[str, float] = {leg: -1e9 for leg in LEG_ORDER}

    row_count = max(len(samples) for samples in samples_by_leg.values())
    for row_index in range(row_count):
        sample_group = {leg: samples[row_index] for leg, samples in samples_by_leg.items() if row_index < len(samples)}
        if not sample_group:
            continue
        timestamp = max(sample.timestamp for sample in sample_group.values())
        result = detector.add_samples(sample_group)
        if result.candidate_score > max_score:
            max_score = result.candidate_score
            max_leg = result.candidate_leg
        if result.alarm_leg and timestamp - last_logged[result.alarm_leg] >= cooldown_seconds:
            last_logged[result.alarm_leg] = timestamp
            events += 1
            append_event(log_path, timestamp, result)
            torque_threshold = detector.models[result.alarm_leg].torque_threshold
            polarity = detector.models[result.alarm_leg].polarity
            print(f"ENTANGLEMENT DETECTED t={timestamp:.3f}s leg={result.alarm_leg} ({LEG_NAMES[result.alarm_leg]}) persistence_score={result.candidate_score:.2f} thigh_tau_{polarity}_threshold={torque_threshold:.3f}")
        elif log_normal and row_index % 50 == 0:
            torque_threshold = detector.models[result.candidate_leg].torque_threshold if result.candidate_leg else 0.0
            polarity = detector.models[result.candidate_leg].polarity if result.candidate_leg else "signed"
            print(f"t={timestamp:.3f}s top={result.candidate_leg} persistence_score={result.candidate_score:.2f} thigh_tau_{polarity}_threshold={torque_threshold:.3f}")

    print(f"Replay finished: {csv_path}")
    print(f"Logged events: {events}")
    print(f"Max score: leg={max_leg} score={max_score:.2f}")


def extract_live_samples(msg: Any, timestamp: float, foot_order: tuple[str, ...]) -> dict[str, TorqueSample]:
    if not hasattr(msg, "motor_state"):
        return {}
    motor_state = list(msg.motor_state)
    if len(motor_state) < 12:
        return {}

    raw_foot = getattr(msg, "foot_force", [])
    try:
        foot_values = [safe_float(value) for value in raw_foot]
    except TypeError:
        foot_values = []
    foot_by_leg = {leg: foot_values[index] if index < len(foot_values) else 0.0 for index, leg in enumerate(foot_order)}

    samples: dict[str, TorqueSample] = {}
    for leg, joint_indices in MOTOR_INDEX.items():
        thigh = motor_state[joint_indices["thigh"]]
        joint_states = [motor_state[joint_indices[joint]] for joint in JOINT_ORDER]
        torque_abs_sum = sum(abs(safe_float(getattr(joint_state, "tau_est", getattr(joint_state, "tau", 0.0)))) for joint_state in joint_states)
        dq_abs_mean = fmean(abs(safe_float(getattr(joint_state, "dq", 0.0))) for joint_state in joint_states)
        samples[leg] = TorqueSample(
            timestamp=timestamp,
            thigh_tau=safe_float(getattr(thigh, "tau_est", getattr(thigh, "tau", 0.0))),
            thigh_dq=safe_float(getattr(thigh, "dq", 0.0)),
            torque_abs_sum=torque_abs_sum,
            dq_abs_mean=dq_abs_mean,
            foot_force=foot_by_leg.get(leg, 0.0),
        )
    return samples


def run_live(args: argparse.Namespace, detector: ThighTorqueDetector, foot_order: tuple[str, ...]) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from unitree_go.msg import LowState
    except ImportError as error:
        raise SystemExit("Run live mode on the robot/ROS2 machine where rclpy and unitree_go are installed.") from error

    last_logged: dict[str, float] = {leg: -1e9 for leg in LEG_ORDER}

    class EntanglementNode(Node):
        def __init__(self) -> None:
            super().__init__("thigh_torque_entanglement_detector")
            self.subscription = self.create_subscription(LowState, "/lowstate", self.callback, 10)
            self.get_logger().info("Listening on /lowstate")

        def callback(self, msg: LowState) -> None:
            timestamp = self.get_clock().now().nanoseconds / 1e9
            sample_group = extract_live_samples(msg, timestamp, foot_order)
            if not sample_group:
                self.get_logger().warn("Could not read motor_state from LowState")
                return

            result = detector.add_samples(sample_group)
            if result.alarm_leg and timestamp - last_logged[result.alarm_leg] >= args.cooldown_seconds:
                last_logged[result.alarm_leg] = timestamp
                append_event(args.log_path, timestamp, result)
                torque_threshold = detector.models[result.alarm_leg].torque_threshold
                polarity = detector.models[result.alarm_leg].polarity
                self.get_logger().warn(
                    f"ENTANGLEMENT DETECTED leg={result.alarm_leg} ({LEG_NAMES[result.alarm_leg]}) "
                    f"persistence_score={result.candidate_score:.2f} thigh_tau_{polarity}_threshold={torque_threshold:.3f} "
                    f"scores={{{', '.join(f'{leg}:{score:.2f}' for leg, score in sorted(result.scores.items()))}}}"
                )
            elif args.log_normal:
                torque_threshold = detector.models[result.candidate_leg].torque_threshold if result.candidate_leg else 0.0
                polarity = detector.models[result.candidate_leg].polarity if result.candidate_leg else "signed"
                self.get_logger().info(f"top={result.candidate_leg} persistence_score={result.candidate_score:.2f} thigh_tau_{polarity}_threshold={torque_threshold:.3f}")

    rclpy.init()
    node = EntanglementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def print_model_summary(models: dict[str, LegModel]) -> None:
    print("Signed thigh-torque thresholds:")
    for leg in LEG_ORDER:
        model = models[leg]
        ent_text = "n/a" if model.entanglement_lower_tau is None else f"{model.entanglement_lower_tau:.3f}"
        print(
            f"  {leg} ({LEG_NAMES[leg]}): polarity={model.polarity}, threshold_tau={model.torque_threshold:.3f}, "
            f"walking_median_tau={model.calibration_median_tau:.3f}, "
            f"walking_lower_tau={model.walking_lower_tau:.3f}, "
            f"walking_upper_tau={model.walking_upper_tau:.3f}, "
            f"entanglement_extreme_tau={ent_text}, "
            f"walking_min_tau={model.calibration_min_tau:.3f}, "
            f"walking_max_tau={model.calibration_max_tau:.3f}"
        )


def main() -> None:
    args = parse_args()
    foot_order = tuple(part.strip().upper() for part in args.foot_force_order.split(",") if part.strip())
    if len(foot_order) != 4 or any(leg not in LEG_ORDER for leg in foot_order):
        raise SystemExit("--foot-force-order must contain exactly four leg labels from FR,FL,RR,RL")

    if args.model_config.exists() and not args.force_calibrate:
        models, config_payload = load_model_config(args.model_config)
        args.window_seconds = safe_float(config_payload.get("window_seconds"), args.window_seconds)
        args.persistence_seconds = safe_float(config_payload.get("persistence_seconds"), args.persistence_seconds)
        args.required_window_fraction = safe_float(config_payload.get("required_window_fraction"), args.required_window_fraction)
        args.cooldown_seconds = safe_float(config_payload.get("cooldown_seconds"), args.cooldown_seconds)
        print(f"Loaded fixed detector parameters from {args.model_config}")
    else:
        entanglement_cases = parse_entanglement_cases(args.entanglement_case)
        models = calibrate_models(
            walking_csv=args.walking_csv,
            entanglement_cases=entanglement_cases,
            calibration_seconds=args.calibration_seconds,
            walking_lower_percentile=args.walking_lower_percentile,
            entanglement_lower_percentile=args.entanglement_lower_percentile,
            threshold_blend=args.threshold_blend,
        )
        if args.save_model_config:
            save_model_config(args.model_config, models, args)
            print(f"Saved fixed detector parameters to {args.model_config}")

    print_model_summary(models)
    print(
        f"Detection rule: front legs alarm on upward thigh_tau spikes and rear legs alarm on downward thigh_tau spikes for >= {args.persistence_seconds:.3f}s "
        f"and >= {args.required_window_fraction:.0%} of the last {args.window_seconds:.3f}s window."
    )
    init_event_log(args.log_path)

    detector = ThighTorqueDetector(
        models=models,
        window_seconds=args.window_seconds,
        persistence_seconds=args.persistence_seconds,
        required_window_fraction=args.required_window_fraction,
    )
    if args.replay_csv is not None:
        replay_csv(args.replay_csv, detector, args.log_path, args.log_normal, args.cooldown_seconds)
        return

    run_live(args, detector, foot_order)


if __name__ == "__main__":
    main()
