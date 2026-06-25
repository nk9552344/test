#!/usr/bin/env python3
"""Replay-only entanglement/stop event logger for Go2 lowstate CSVs.

Rule implemented from observed data:
  - Monitor both rear thigh torques: RR and RL.
  - Sustained rear-leg downward spike means back-leg entanglement.
  - Sustained rear-leg upward spike means front-leg entanglement.
  - Stop is detected from very large short negative front thigh-torque spikes,
    especially FL, with only a few spike samples in the window.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Deque, Iterable


REAR_LEGS = ("RR", "RL")
FRONT_LEGS = ("FR", "FL")
LEG_NAMES = {
    "FL": "front_left",
    "FR": "front_right",
    "RL": "back_left",
    "RR": "back_right",
}

DEFAULT_WALKING_CSV = "go2_lowstate_walking_v2.csv"
DEFAULT_ENTANGLEMENT_CSV = "go2_lowstate_back_right_leg.csv"
DEFAULT_STOPS_CSV = "go2_lowstate_stops.csv"
THIGH_MOTOR_INDEX = {
    "FR": 1,
    "FL": 4,
    "RR": 7,
    "RL": 10,
}


@dataclass(frozen=True)
class Sample:
    timestamp: float
    values: dict[str, float]


@dataclass(frozen=True)
class LegLimits:
    center: float
    down_threshold: float
    up_threshold: float
    down_change_threshold: float
    up_change_threshold: float
    scale: float


@dataclass(frozen=True)
class StopLimits:
    threshold: float
    scale: float
    max_spike_fraction: float


@dataclass(frozen=True)
class Calibration:
    rear_limits: dict[str, LegLimits]
    stop_limits: dict[str, StopLimits]


@dataclass(frozen=True)
class RearSpike:
    leg: str
    direction: str
    score: float
    extreme_tau: float
    fraction: float
    duration: float


@dataclass(frozen=True)
class StopSpike:
    leg: str
    score: float
    extreme_tau: float
    spike_count: int
    fraction: float


@dataclass(frozen=True)
class WindowResult:
    timestamp: float
    rear_spike: RearSpike | None
    stop_spike: StopSpike | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Go2 lowstate CSV and print entanglement/stop events.")
    parser.add_argument("--live", action="store_true", help="Subscribe to /lowstate instead of replaying a CSV.")
    parser.add_argument("--topic", default="/lowstate", help="ROS2 LowState topic used in --live mode.")
    parser.add_argument("--replay-csv", type=Path, default=None, help="CSV to replay. Defaults to back-left entanglement data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder containing Go2 CSV files.")
    parser.add_argument("--walking-csv", type=Path, default=None, help="Normal walking baseline CSV.")
    parser.add_argument("--stops-csv", type=Path, default=None, help="Random-stop calibration CSV.")
    parser.add_argument("--entanglement-csv", type=Path, default=None, help="Optional labeled replay CSV. Not used for rear threshold calibration.")
    parser.add_argument("--entanglement-start", type=float, default=4.0, help="Optional label start time, kept for compatibility.")
    parser.add_argument("--entanglement-end", type=float, default=8.0, help="Optional label end time, kept for compatibility.")
    parser.add_argument("--window-seconds", type=float, default=0.80, help="Rolling window length.")
    parser.add_argument("--persistence-seconds", type=float, default=0.30, help="Abnormal rear samples must span at least this much time in the window.")
    parser.add_argument("--required-window-fraction", type=float, default=0.25, help="Rear spike fraction required in the window.")
    parser.add_argument("--rear-sigma", type=float, default=4.0, help="Walking baseline sigma multiplier for rear spike thresholds.")
    parser.add_argument("--rear-down-threshold", type=float, default=-5.2, help="Rear thigh_tau at or below this sustained value means back entanglement.")
    parser.add_argument("--rear-up-threshold", type=float, default=11.0, help="Rear thigh_tau at or above this sustained value means front entanglement.")
    parser.add_argument("--stop-threshold", type=float, default=-16.0, help="Front thigh_tau below this is considered a stop spike.")
    parser.add_argument("--stop-max-spike-fraction", type=float, default=0.22, help="Stops have only a few front spike samples in the window.")
    parser.add_argument("--stop-reset-seconds", type=float, default=0.80, help="After STOP, clear the rolling window and ignore rear entanglement for this long.")
    parser.add_argument("--stop-lookback-seconds", type=float, default=0.60, help="In replay mode, suppress rear events this long before a detected STOP.")
    parser.add_argument("--cooldown-seconds", type=float, default=0.80, help="Minimum time between printed events of same type.")
    parser.add_argument("--summary-only", action="store_true", help="Print event lines and final summary only.")
    parser.add_argument("--log-normal", action="store_true", help="Print periodic non-event status lines.")
    return parser.parse_args()


def safe_float(raw: str | None, default: float = 0.0) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def percentile(values: Iterable[float], q: float) -> float:
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


def median(values: Iterable[float]) -> float:
    return percentile(values, 50.0)


def robust_scale(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return 1.0
    center = median(clean)
    deviations = [abs(value - center) for value in clean]
    mad_sigma = 1.4826 * median(deviations)
    if mad_sigma > 1e-6:
        return mad_sigma
    if len(clean) > 1:
        mean_value = fmean(clean)
        variance = fmean((value - mean_value) ** 2 for value in clean)
        return max(math.sqrt(variance), 1e-6)
    return 1.0


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_data_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    base = script_dir()
    candidates = (
        base / "data",
        base / "data_analysis" / "data",
        base.parent / "data",
        Path.cwd() / "data",
    )
    for candidate in candidates:
        if (candidate / DEFAULT_WALKING_CSV).exists():
            return candidate
    return candidates[0]


def resolve_csv(path: Path | None, data_dir: Path, default_name: str) -> Path:
    if path is None:
        return data_dir / default_name
    if path.exists() or path.is_absolute():
        return path
    data_path = data_dir / path
    if data_path.exists():
        return data_path
    named_data_path = data_dir / path.name
    if named_data_path.exists():
        return named_data_path
    return path


def read_samples(csv_path: Path) -> list[Sample]:
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    required = ["timestamp"] + [f"{leg}_thigh_tau" for leg in FRONT_LEGS + REAR_LEGS]
    samples: list[Sample] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")

        first_timestamp: float | None = None
        for row in reader:
            raw_timestamp = safe_float(row.get("timestamp"))
            if first_timestamp is None:
                first_timestamp = raw_timestamp
            values = {column: safe_float(row.get(column)) for column in required if column != "timestamp"}
            samples.append(Sample(timestamp=raw_timestamp - first_timestamp, values=values))

    if not samples:
        raise SystemExit(f"CSV contains no data rows: {csv_path}")
    return samples


def walking_change_threshold(samples: list[Sample], leg: str, window_seconds: float) -> float:
    history: Deque[Sample] = deque()
    ranges: list[float] = []
    column = f"{leg}_thigh_tau"
    for sample in samples:
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) < 2:
            continue
        values = [history_sample.values[column] for history_sample in history]
        ranges.append(max(values) - min(values))
    if not ranges:
        return 1.0
    return max(percentile(ranges, 99.5), robust_scale(ranges), 1e-6)


def rear_window_values(samples: list[Sample], start: float, end: float) -> list[float]:
    values: list[float] = []
    for sample in samples:
        if start <= sample.timestamp <= end:
            values.extend(sample.values[f"{leg}_thigh_tau"] for leg in REAR_LEGS)
    return values


def calibrate(
    walking: list[Sample],
    stops: list[Sample],
    entanglement: list[Sample],
    entanglement_start: float,
    entanglement_end: float,
    window_seconds: float,
    rear_sigma: float,
    rear_down_threshold: float,
    rear_up_threshold: float,
    stop_threshold: float,
    stop_max_spike_fraction: float,
) -> Calibration:
    rear_limits: dict[str, LegLimits] = {}
    labeled_rear_values = rear_window_values(entanglement, entanglement_start, entanglement_end)
    labeled_low = percentile(labeled_rear_values, 20.0) if labeled_rear_values else None
    labeled_high = percentile(labeled_rear_values, 80.0) if labeled_rear_values else None
    for leg in REAR_LEGS:
        values = [sample.values[f"{leg}_thigh_tau"] for sample in walking]
        center = median(values)
        scale = robust_scale(values)
        lower_guard = percentile(values, 0.5)
        upper_guard = percentile(values, 99.5)
        down_threshold = rear_down_threshold
        up_threshold = rear_up_threshold
        if labeled_low is not None and labeled_low < lower_guard:
            down_threshold = max(rear_down_threshold, (lower_guard + labeled_low) / 2.0)
        if labeled_high is not None and labeled_high > upper_guard:
            up_threshold = min(rear_up_threshold, (upper_guard + labeled_high) / 2.0)
        change_threshold = walking_change_threshold(walking, leg, window_seconds)
        rear_limits[leg] = LegLimits(
            center=center,
            down_threshold=down_threshold,
            up_threshold=up_threshold,
            down_change_threshold=change_threshold,
            up_change_threshold=change_threshold,
            scale=max(scale, 1e-6),
        )

    stop_limits: dict[str, StopLimits] = {}
    for leg in FRONT_LEGS:
        walking_values = [sample.values[f"{leg}_thigh_tau"] for sample in walking]
        stops_values = [sample.values[f"{leg}_thigh_tau"] for sample in stops]
        learned_threshold = percentile(stops_values, 2.0) if stops_values else stop_threshold
        threshold = min(stop_threshold, (percentile(walking_values, 0.5) + learned_threshold) / 2.0)
        stop_limits[leg] = StopLimits(
            threshold=threshold,
            scale=max(abs(percentile(walking_values, 0.5) - threshold), robust_scale(walking_values), 1e-6),
            max_spike_fraction=stop_max_spike_fraction,
        )

    return Calibration(rear_limits=rear_limits, stop_limits=stop_limits)


def latest_continuous_duration(samples: list[Sample], column: str, threshold: float, direction: str) -> float:
    if len(samples) < 2:
        return 0.0

    def exceeds(sample: Sample) -> bool:
        value = sample.values[column]
        return value <= threshold if direction == "down" else value >= threshold

    if not exceeds(samples[-1]):
        return 0.0
    start_time = samples[-1].timestamp
    for sample in reversed(samples[:-1]):
        if not exceeds(sample):
            break
        start_time = sample.timestamp
    return samples[-1].timestamp - start_time


def abnormal_span(samples: list[Sample], flags: list[bool]) -> float:
    abnormal_times = [sample.timestamp for sample, flag in zip(samples, flags) if flag]
    if len(abnormal_times) < 2:
        return 0.0
    return abnormal_times[-1] - abnormal_times[0]


def score_rear_leg(samples: list[Sample], leg: str, limits: LegLimits, required_fraction: float, persistence_seconds: float) -> RearSpike | None:
    values = [sample.values[f"{leg}_thigh_tau"] for sample in samples]
    window_high = max(values)
    window_low = min(values)
    down_flags = [value <= limits.down_threshold or window_high - value >= limits.down_change_threshold for value in values]
    up_flags = [value >= limits.up_threshold or value - window_low >= limits.up_change_threshold for value in values]
    down_count = sum(1 for flag in down_flags if flag)
    up_count = sum(1 for flag in up_flags if flag)
    down_fraction = down_count / len(values)
    up_fraction = up_count / len(values)

    candidates: list[RearSpike] = []
    if down_fraction >= required_fraction:
        extreme = min(values)
        duration = latest_continuous_duration(samples, f"{leg}_thigh_tau", limits.down_threshold, "down")
        duration = max(duration, abnormal_span(samples, down_flags))
        if duration < persistence_seconds and window_high - extreme >= limits.down_change_threshold:
            duration = samples[-1].timestamp - samples[0].timestamp
        if duration >= persistence_seconds:
            abs_score = 1.0 + max(0.0, limits.down_threshold - extreme) / limits.scale
            change_score = 1.0 + max(0.0, window_high - extreme - limits.down_change_threshold) / limits.scale
            level_score = max(abs_score, change_score)
            candidates.append(RearSpike(leg, "down", level_score, extreme, down_fraction, duration))
    if up_fraction >= required_fraction:
        extreme = max(values)
        duration = latest_continuous_duration(samples, f"{leg}_thigh_tau", limits.up_threshold, "up")
        duration = max(duration, abnormal_span(samples, up_flags))
        if duration < persistence_seconds and extreme - window_low >= limits.up_change_threshold:
            duration = samples[-1].timestamp - samples[0].timestamp
        if duration >= persistence_seconds:
            abs_score = 1.0 + max(0.0, extreme - limits.up_threshold) / limits.scale
            change_score = 1.0 + max(0.0, extreme - window_low - limits.up_change_threshold) / limits.scale
            level_score = max(abs_score, change_score)
            candidates.append(RearSpike(leg, "up", level_score, extreme, up_fraction, duration))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def detect_stop(samples: list[Sample], calibration: Calibration) -> StopSpike | None:
    candidates: list[StopSpike] = []
    for leg in FRONT_LEGS:
        limits = calibration.stop_limits[leg]
        values = [sample.values[f"{leg}_thigh_tau"] for sample in samples]
        spike_values = [value for value in values if value <= limits.threshold]
        if not spike_values:
            continue
        fraction = len(spike_values) / len(values)
        if fraction > limits.max_spike_fraction:
            continue
        extreme = min(spike_values)
        score = 1.0 + (limits.threshold - extreme) / limits.scale
        candidates.append(StopSpike(leg, score, extreme, len(spike_values), fraction))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def evaluate_window(history: Deque[Sample], calibration: Calibration, required_fraction: float, persistence_seconds: float) -> WindowResult:
    samples = list(history)
    rear_candidates = [
        candidate
        for leg in REAR_LEGS
        if (candidate := score_rear_leg(samples, leg, calibration.rear_limits[leg], required_fraction, persistence_seconds)) is not None
    ]
    rear_spike = max(rear_candidates, key=lambda candidate: candidate.score) if rear_candidates else None
    stop_spike = detect_stop(samples, calibration)
    return WindowResult(timestamp=samples[-1].timestamp, rear_spike=rear_spike, stop_spike=stop_spike)


def event_type(rear_spike: RearSpike) -> str:
    return "ENTANGLE_BACK" if rear_spike.direction == "down" else "ENTANGLE_FRONT"


def stop_suppression_intervals(
    samples: list[Sample],
    calibration: Calibration,
    window_seconds: float,
    stop_lookback_seconds: float,
    stop_reset_seconds: float,
) -> list[tuple[float, float]]:
    history: Deque[Sample] = deque()
    intervals: list[tuple[float, float]] = []
    for sample in samples:
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) < 2:
            continue
        if detect_stop(list(history), calibration) is not None:
            intervals.append((sample.timestamp - stop_lookback_seconds, sample.timestamp + stop_reset_seconds))

    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def in_intervals(timestamp: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= timestamp <= end for start, end in intervals)


def print_calibration(calibration: Calibration, walking_csv: Path, stops_csv: Path, entanglement_csv: Path) -> None:
    print("Calibration")
    print("-----------")
    print(f"Walking baseline     : {walking_csv}")
    print(f"Stop baseline        : {stops_csv}")
    print(f"Entanglement baseline: {entanglement_csv}")
    print("Rear thigh torque rules")
    for leg in REAR_LEGS:
        limits = calibration.rear_limits[leg]
        print(
            f"  {leg}/{LEG_NAMES[leg]} down<={limits.down_threshold:.3f} => ENTANGLE_BACK, "
            f"up>={limits.up_threshold:.3f} => ENTANGLE_FRONT, "
            f"change>={limits.down_change_threshold:.3f}"
        )
    print("Stop rules")
    for leg in FRONT_LEGS:
        limits = calibration.stop_limits[leg]
        print(f"  {leg}/{LEG_NAMES[leg]} thigh_tau<={limits.threshold:.3f}, spike_fraction<={limits.max_spike_fraction:.2f} => STOP")
    print()


def replay(
    samples: list[Sample],
    calibration: Calibration,
    window_seconds: float,
    persistence_seconds: float,
    required_fraction: float,
    stop_reset_seconds: float,
    stop_lookback_seconds: float,
    cooldown_seconds: float,
    summary_only: bool,
    log_normal: bool,
) -> None:
    history: Deque[Sample] = deque()
    stop_intervals = stop_suppression_intervals(samples, calibration, window_seconds, stop_lookback_seconds, stop_reset_seconds)
    stop_reset_until = -1e9
    last_by_type: dict[str, float] = {"ENTANGLE_BACK": -1e9, "ENTANGLE_FRONT": -1e9, "STOP": -1e9}
    counts = {"ENTANGLE_BACK": 0, "ENTANGLE_FRONT": 0, "STOP": 0}
    first_events: dict[str, float | None] = {"ENTANGLE_BACK": None, "ENTANGLE_FRONT": None, "STOP": None}
    reset_windows = 0

    for row_index, sample in enumerate(samples):
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) < 2:
            continue

        result = evaluate_window(history, calibration, required_fraction, persistence_seconds)
        stop_active = result.stop_spike is not None
        if stop_active and result.timestamp - last_by_type["STOP"] >= cooldown_seconds:
            stop = result.stop_spike
            last_by_type["STOP"] = result.timestamp
            counts["STOP"] += 1
            first_events["STOP"] = first_events["STOP"] if first_events["STOP"] is not None else result.timestamp
            print(
                f"t={result.timestamp:.3f}s type=STOP leg={stop.leg}/{LEG_NAMES[stop.leg]} "
                f"score={stop.score:.2f} tau={stop.extreme_tau:.3f} spikes={stop.spike_count} frac={stop.fraction:.2f}"
            )
        if stop_active:
            stop_reset_until = max(stop_reset_until, result.timestamp + stop_reset_seconds)
            history.clear()
            continue

        if result.timestamp < stop_reset_until:
            reset_windows += 1
            history.clear()
            continue

        if result.rear_spike is not None and not in_intervals(result.timestamp, stop_intervals):
            rear = result.rear_spike
            kind = event_type(rear)
            if result.timestamp - last_by_type[kind] >= cooldown_seconds:
                last_by_type[kind] = result.timestamp
                counts[kind] += 1
                first_events[kind] = first_events[kind] if first_events[kind] is not None else result.timestamp
                print(
                    f"t={result.timestamp:.3f}s type={kind} rear_leg={rear.leg}/{LEG_NAMES[rear.leg]} "
                    f"dir={rear.direction} score={rear.score:.2f} tau={rear.extreme_tau:.3f} "
                    f"frac={rear.fraction:.2f} dur={rear.duration:.3f}s"
                )
        elif log_normal and not summary_only and row_index % 50 == 0:
            print(f"t={result.timestamp:.3f}s type=NORMAL")

    print()
    print("Summary")
    print("-------")
    print(f"Rows replayed        : {len(samples)}")
    print(f"ENTANGLE_BACK events : {counts['ENTANGLE_BACK']}")
    print(f"ENTANGLE_FRONT events: {counts['ENTANGLE_FRONT']}")
    print(f"STOP events          : {counts['STOP']}")
    print(f"Stop-reset windows   : {reset_windows}")
    print(f"Stop-veto intervals  : {len(stop_intervals)}")
    for kind in ("ENTANGLE_BACK", "ENTANGLE_FRONT", "STOP"):
        first = first_events[kind]
        first_text = "none" if first is None else f"t={first:.3f}s"
        print(f"First {kind:<14}: {first_text}")


def sample_from_lowstate(msg: object, timestamp: float) -> Sample | None:
    if not hasattr(msg, "motor_state"):
        return None
    motor_state = list(getattr(msg, "motor_state"))
    if len(motor_state) <= max(THIGH_MOTOR_INDEX.values()):
        return None

    values: dict[str, float] = {}
    for leg, index in THIGH_MOTOR_INDEX.items():
        joint = motor_state[index]
        values[f"{leg}_thigh_tau"] = safe_float(getattr(joint, "tau_est", getattr(joint, "tau", 0.0)))
    return Sample(timestamp=timestamp, values=values)


def run_live(
    calibration: Calibration,
    topic: str,
    window_seconds: float,
    persistence_seconds: float,
    required_fraction: float,
    stop_reset_seconds: float,
    cooldown_seconds: float,
    log_normal: bool,
) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from unitree_go.msg import LowState
    except ImportError as error:
        raise SystemExit("Live mode requires ROS2 rclpy and unitree_go.msg.LowState on the robot machine.") from error

    class LiveDetector(Node):
        def __init__(self) -> None:
            super().__init__("rear_thigh_entanglement_test_script")
            self.history: Deque[Sample] = deque()
            self.start_time: float | None = None
            self.stop_reset_until = -1e9
            self.last_by_type = {"ENTANGLE_BACK": -1e9, "ENTANGLE_FRONT": -1e9, "STOP": -1e9}
            self.subscription = self.create_subscription(LowState, topic, self.callback, 10)
            self.get_logger().info(f"Listening on {topic}")

        def callback(self, msg: object) -> None:
            now = self.get_clock().now().nanoseconds / 1e9
            if self.start_time is None:
                self.start_time = now
            sample = sample_from_lowstate(msg, now - self.start_time)
            if sample is None:
                self.get_logger().warn("LowState message does not contain usable motor_state thigh torque values")
                return

            self.history.append(sample)
            cutoff = sample.timestamp - window_seconds
            while self.history and self.history[0].timestamp < cutoff:
                self.history.popleft()
            if len(self.history) < 2:
                return

            result = evaluate_window(self.history, calibration, required_fraction, persistence_seconds)
            stop_active = result.stop_spike is not None
            if stop_active and result.timestamp - self.last_by_type["STOP"] >= cooldown_seconds:
                stop = result.stop_spike
                self.last_by_type["STOP"] = result.timestamp
                self.get_logger().warn(
                    f"t={result.timestamp:.3f}s type=STOP leg={stop.leg}/{LEG_NAMES[stop.leg]} "
                    f"score={stop.score:.2f} tau={stop.extreme_tau:.3f} spikes={stop.spike_count} frac={stop.fraction:.2f}"
                )
            if stop_active:
                self.stop_reset_until = max(self.stop_reset_until, result.timestamp + stop_reset_seconds)
                self.history.clear()
                return

            if result.timestamp < self.stop_reset_until:
                self.history.clear()
                return

            if result.rear_spike is not None:
                rear = result.rear_spike
                kind = event_type(rear)
                if result.timestamp - self.last_by_type[kind] >= cooldown_seconds:
                    self.last_by_type[kind] = result.timestamp
                    self.get_logger().warn(
                        f"t={result.timestamp:.3f}s type={kind} rear_leg={rear.leg}/{LEG_NAMES[rear.leg]} "
                        f"dir={rear.direction} score={rear.score:.2f} tau={rear.extreme_tau:.3f} "
                        f"frac={rear.fraction:.2f} dur={rear.duration:.3f}s"
                    )
            elif log_normal:
                self.get_logger().info(f"t={result.timestamp:.3f}s type=NORMAL")

    rclpy.init()
    node = LiveDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    walking_csv = resolve_csv(args.walking_csv, data_dir, DEFAULT_WALKING_CSV)
    stops_csv = resolve_csv(args.stops_csv, data_dir, DEFAULT_STOPS_CSV)
    entanglement_csv = resolve_csv(args.entanglement_csv, data_dir, DEFAULT_ENTANGLEMENT_CSV)
    replay_csv = resolve_csv(args.replay_csv, data_dir, DEFAULT_ENTANGLEMENT_CSV)

    walking = read_samples(walking_csv)
    stops = read_samples(stops_csv)
    entanglement = read_samples(entanglement_csv)
    calibration = calibrate(
        walking,
        stops,
        entanglement,
        args.entanglement_start,
        args.entanglement_end,
        args.window_seconds,
        args.rear_sigma,
        args.rear_down_threshold,
        args.rear_up_threshold,
        args.stop_threshold,
        args.stop_max_spike_fraction,
    )
    if not args.summary_only:
        print_calibration(calibration, walking_csv, stops_csv, entanglement_csv)
        if args.live:
            print(f"Live topic       : {args.topic}")
        else:
            print(f"Replay CSV       : {replay_csv}")
        print()

    if args.live:
        run_live(
            calibration=calibration,
            topic=args.topic,
            window_seconds=args.window_seconds,
            persistence_seconds=args.persistence_seconds,
            required_fraction=args.required_window_fraction,
            stop_reset_seconds=args.stop_reset_seconds,
            cooldown_seconds=args.cooldown_seconds,
            log_normal=args.log_normal,
        )
        return

    replay(
        samples=read_samples(replay_csv),
        calibration=calibration,
        window_seconds=args.window_seconds,
        persistence_seconds=args.persistence_seconds,
        required_fraction=args.required_window_fraction,
        stop_reset_seconds=args.stop_reset_seconds,
        stop_lookback_seconds=args.stop_lookback_seconds,
        cooldown_seconds=args.cooldown_seconds,
        summary_only=args.summary_only,
        log_normal=args.log_normal,
    )


if __name__ == "__main__":
    main()
'''
#!/usr/bin/env python3
"""Replay-only entanglement/stop event logger for Go2 lowstate CSVs.

Rule implemented from observed data:
  - Monitor both rear thigh torques: RR and RL.
  - Sustained rear-leg downward spike means back-leg entanglement.
  - Sustained rear-leg upward spike means front-leg entanglement.
  - Stop is detected from very large short negative front thigh-torque spikes,
    especially FL, with only a few spike samples in the window.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Deque, Iterable


REAR_LEGS = ("RR", "RL")
FRONT_LEGS = ("FR", "FL")
LEG_NAMES = {
    "FL": "front_left",
    "FR": "front_right",
    "RL": "back_left",
    "RR": "back_right",
}

DEFAULT_WALKING_CSV = "go2_lowstate_walking_v2.csv"
DEFAULT_ENTANGLEMENT_CSV = "go2_lowstate_back_left_leg.csv"
DEFAULT_STOPS_CSV = "go2_lowstate_stops.csv"


@dataclass(frozen=True)
class Sample:
    timestamp: float
    values: dict[str, float]


@dataclass(frozen=True)
class LegLimits:
    center: float
    down_threshold: float
    up_threshold: float
    scale: float


@dataclass(frozen=True)
class StopLimits:
    threshold: float
    scale: float
    max_spike_fraction: float


@dataclass(frozen=True)
class Calibration:
    rear_limits: dict[str, LegLimits]
    stop_limits: dict[str, StopLimits]


@dataclass(frozen=True)
class RearSpike:
    leg: str
    direction: str
    score: float
    extreme_tau: float
    fraction: float
    duration: float


@dataclass(frozen=True)
class StopSpike:
    leg: str
    score: float
    extreme_tau: float
    spike_count: int
    fraction: float


@dataclass(frozen=True)
class WindowResult:
    timestamp: float
    rear_spike: RearSpike | None
    stop_spike: StopSpike | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a Go2 lowstate CSV and print entanglement/stop events.")
    parser.add_argument("--replay-csv", type=Path, default=None, help="CSV to replay. Defaults to back-left entanglement data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder containing Go2 CSV files.")
    parser.add_argument("--walking-csv", type=Path, default=None, help="Normal walking baseline CSV.")
    parser.add_argument("--stops-csv", type=Path, default=None, help="Random-stop calibration CSV.")
    parser.add_argument("--window-seconds", type=float, default=0.35, help="Rolling window length.")
    parser.add_argument("--persistence-seconds", type=float, default=0.20, help="Continuous rear spike duration required.")
    parser.add_argument("--required-window-fraction", type=float, default=0.55, help="Rear spike fraction required in the window.")
    parser.add_argument("--rear-sigma", type=float, default=4.0, help="Walking baseline sigma multiplier for rear spike thresholds.")
    parser.add_argument("--stop-threshold", type=float, default=-16.0, help="Front thigh_tau below this is considered a stop spike.")
    parser.add_argument("--stop-max-spike-fraction", type=float, default=0.22, help="Stops have only a few front spike samples in the window.")
    parser.add_argument("--cooldown-seconds", type=float, default=0.80, help="Minimum time between printed events of same type.")
    parser.add_argument("--summary-only", action="store_true", help="Print event lines and final summary only.")
    parser.add_argument("--log-normal", action="store_true", help="Print periodic non-event status lines.")
    return parser.parse_args()


def safe_float(raw: str | None, default: float = 0.0) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def percentile(values: Iterable[float], q: float) -> float:
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


def median(values: Iterable[float]) -> float:
    return percentile(values, 50.0)


def robust_scale(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return 1.0
    center = median(clean)
    deviations = [abs(value - center) for value in clean]
    mad_sigma = 1.4826 * median(deviations)
    if mad_sigma > 1e-6:
        return mad_sigma
    if len(clean) > 1:
        mean_value = fmean(clean)
        variance = fmean((value - mean_value) ** 2 for value in clean)
        return max(math.sqrt(variance), 1e-6)
    return 1.0


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_data_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    base = script_dir()
    candidates = (
        base / "data",
        base / "data_analysis" / "data",
        base.parent / "data",
        Path.cwd() / "data",
    )
    for candidate in candidates:
        if (candidate / DEFAULT_WALKING_CSV).exists():
            return candidate
    return candidates[0]


def resolve_csv(path: Path | None, data_dir: Path, default_name: str) -> Path:
    if path is None:
        return data_dir / default_name
    if path.exists() or path.is_absolute():
        return path
    data_path = data_dir / path
    if data_path.exists():
        return data_path
    named_data_path = data_dir / path.name
    if named_data_path.exists():
        return named_data_path
    return path


def read_samples(csv_path: Path) -> list[Sample]:
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    required = ["timestamp"] + [f"{leg}_thigh_tau" for leg in FRONT_LEGS + REAR_LEGS]
    samples: list[Sample] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")

        first_timestamp: float | None = None
        for row in reader:
            raw_timestamp = safe_float(row.get("timestamp"))
            if first_timestamp is None:
                first_timestamp = raw_timestamp
            values = {column: safe_float(row.get(column)) for column in required if column != "timestamp"}
            samples.append(Sample(timestamp=raw_timestamp - first_timestamp, values=values))

    if not samples:
        raise SystemExit(f"CSV contains no data rows: {csv_path}")
    return samples


def calibrate(walking: list[Sample], stops: list[Sample], rear_sigma: float, stop_threshold: float, stop_max_spike_fraction: float) -> Calibration:
    rear_limits: dict[str, LegLimits] = {}
    for leg in REAR_LEGS:
        values = [sample.values[f"{leg}_thigh_tau"] for sample in walking]
        center = median(values)
        scale = robust_scale(values)
        lower_guard = percentile(values, 0.5)
        upper_guard = percentile(values, 99.5)
        rear_limits[leg] = LegLimits(
            center=center,
            down_threshold=min(lower_guard, center - rear_sigma * scale),
            up_threshold=max(upper_guard, center + rear_sigma * scale),
            scale=max(scale, 1e-6),
        )

    stop_limits: dict[str, StopLimits] = {}
    for leg in FRONT_LEGS:
        walking_values = [sample.values[f"{leg}_thigh_tau"] for sample in walking]
        stops_values = [sample.values[f"{leg}_thigh_tau"] for sample in stops]
        learned_threshold = percentile(stops_values, 2.0) if stops_values else stop_threshold
        threshold = min(stop_threshold, (percentile(walking_values, 0.5) + learned_threshold) / 2.0)
        stop_limits[leg] = StopLimits(
            threshold=threshold,
            scale=max(abs(percentile(walking_values, 0.5) - threshold), robust_scale(walking_values), 1e-6),
            max_spike_fraction=stop_max_spike_fraction,
        )

    return Calibration(rear_limits=rear_limits, stop_limits=stop_limits)


def latest_continuous_duration(samples: list[Sample], column: str, threshold: float, direction: str) -> float:
    if len(samples) < 2:
        return 0.0

    def exceeds(sample: Sample) -> bool:
        value = sample.values[column]
        return value <= threshold if direction == "down" else value >= threshold

    if not exceeds(samples[-1]):
        return 0.0
    start_time = samples[-1].timestamp
    for sample in reversed(samples[:-1]):
        if not exceeds(sample):
            break
        start_time = sample.timestamp
    return samples[-1].timestamp - start_time


def score_rear_leg(samples: list[Sample], leg: str, limits: LegLimits, required_fraction: float, persistence_seconds: float) -> RearSpike | None:
    values = [sample.values[f"{leg}_thigh_tau"] for sample in samples]
    down_count = sum(1 for value in values if value <= limits.down_threshold)
    up_count = sum(1 for value in values if value >= limits.up_threshold)
    down_fraction = down_count / len(values)
    up_fraction = up_count / len(values)

    candidates: list[RearSpike] = []
    if down_fraction >= required_fraction:
        extreme = min(values)
        duration = latest_continuous_duration(samples, f"{leg}_thigh_tau", limits.down_threshold, "down")
        if duration >= persistence_seconds:
            level_score = 1.0 + (limits.down_threshold - extreme) / limits.scale
            candidates.append(RearSpike(leg, "down", level_score, extreme, down_fraction, duration))
    if up_fraction >= required_fraction:
        extreme = max(values)
        duration = latest_continuous_duration(samples, f"{leg}_thigh_tau", limits.up_threshold, "up")
        if duration >= persistence_seconds:
            level_score = 1.0 + (extreme - limits.up_threshold) / limits.scale
            candidates.append(RearSpike(leg, "up", level_score, extreme, up_fraction, duration))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def detect_stop(samples: list[Sample], calibration: Calibration) -> StopSpike | None:
    candidates: list[StopSpike] = []
    for leg in FRONT_LEGS:
        limits = calibration.stop_limits[leg]
        values = [sample.values[f"{leg}_thigh_tau"] for sample in samples]
        spike_values = [value for value in values if value <= limits.threshold]
        if not spike_values:
            continue
        fraction = len(spike_values) / len(values)
        if fraction > limits.max_spike_fraction:
            continue
        extreme = min(spike_values)
        score = 1.0 + (limits.threshold - extreme) / limits.scale
        candidates.append(StopSpike(leg, score, extreme, len(spike_values), fraction))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def evaluate_window(history: Deque[Sample], calibration: Calibration, required_fraction: float, persistence_seconds: float) -> WindowResult:
    samples = list(history)
    rear_candidates = [
        candidate
        for leg in REAR_LEGS
        if (candidate := score_rear_leg(samples, leg, calibration.rear_limits[leg], required_fraction, persistence_seconds)) is not None
    ]
    rear_spike = max(rear_candidates, key=lambda candidate: candidate.score) if rear_candidates else None
    stop_spike = detect_stop(samples, calibration)
    return WindowResult(timestamp=samples[-1].timestamp, rear_spike=rear_spike, stop_spike=stop_spike)


def event_type(rear_spike: RearSpike) -> str:
    return "ENTANGLE_BACK" if rear_spike.direction == "down" else "ENTANGLE_FRONT"


def print_calibration(calibration: Calibration, walking_csv: Path, stops_csv: Path) -> None:
    print("Calibration")
    print("-----------")
    print(f"Walking baseline : {walking_csv}")
    print(f"Stop baseline    : {stops_csv}")
    print("Rear thigh torque rules")
    for leg in REAR_LEGS:
        limits = calibration.rear_limits[leg]
        print(
            f"  {leg}/{LEG_NAMES[leg]} down<={limits.down_threshold:.3f} => ENTANGLE_BACK, "
            f"up>={limits.up_threshold:.3f} => ENTANGLE_FRONT"
        )
    print("Stop rules")
    for leg in FRONT_LEGS:
        limits = calibration.stop_limits[leg]
        print(f"  {leg}/{LEG_NAMES[leg]} thigh_tau<={limits.threshold:.3f}, spike_fraction<={limits.max_spike_fraction:.2f} => STOP")
    print()


def replay(
    samples: list[Sample],
    calibration: Calibration,
    window_seconds: float,
    persistence_seconds: float,
    required_fraction: float,
    cooldown_seconds: float,
    summary_only: bool,
    log_normal: bool,
) -> None:
    history: Deque[Sample] = deque()
    last_by_type: dict[str, float] = {"ENTANGLE_BACK": -1e9, "ENTANGLE_FRONT": -1e9, "STOP": -1e9}
    counts = {"ENTANGLE_BACK": 0, "ENTANGLE_FRONT": 0, "STOP": 0}
    first_events: dict[str, float | None] = {"ENTANGLE_BACK": None, "ENTANGLE_FRONT": None, "STOP": None}

    for row_index, sample in enumerate(samples):
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) < 2:
            continue

        result = evaluate_window(history, calibration, required_fraction, persistence_seconds)
        stop_active = result.stop_spike is not None
        if stop_active and result.timestamp - last_by_type["STOP"] >= cooldown_seconds:
            stop = result.stop_spike
            last_by_type["STOP"] = result.timestamp
            counts["STOP"] += 1
            first_events["STOP"] = first_events["STOP"] if first_events["STOP"] is not None else result.timestamp
            print(
                f"t={result.timestamp:.3f}s type=STOP leg={stop.leg}/{LEG_NAMES[stop.leg]} "
                f"score={stop.score:.2f} tau={stop.extreme_tau:.3f} spikes={stop.spike_count} frac={stop.fraction:.2f}"
            )

        if result.rear_spike is not None and not stop_active:
            rear = result.rear_spike
            kind = event_type(rear)
            if result.timestamp - last_by_type[kind] >= cooldown_seconds:
                last_by_type[kind] = result.timestamp
                counts[kind] += 1
                first_events[kind] = first_events[kind] if first_events[kind] is not None else result.timestamp
                print(
                    f"t={result.timestamp:.3f}s type={kind} rear_leg={rear.leg}/{LEG_NAMES[rear.leg]} "
                    f"dir={rear.direction} score={rear.score:.2f} tau={rear.extreme_tau:.3f} "
                    f"frac={rear.fraction:.2f} dur={rear.duration:.3f}s"
                )
        elif log_normal and not summary_only and row_index % 50 == 0:
            print(f"t={result.timestamp:.3f}s type=NORMAL")

    print()
    print("Summary")
    print("-------")
    print(f"Rows replayed        : {len(samples)}")
    print(f"ENTANGLE_BACK events : {counts['ENTANGLE_BACK']}")
    print(f"ENTANGLE_FRONT events: {counts['ENTANGLE_FRONT']}")
    print(f"STOP events          : {counts['STOP']}")
    for kind in ("ENTANGLE_BACK", "ENTANGLE_FRONT", "STOP"):
        first = first_events[kind]
        first_text = "none" if first is None else f"t={first:.3f}s"
        print(f"First {kind:<14}: {first_text}")


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    walking_csv = resolve_csv(args.walking_csv, data_dir, DEFAULT_WALKING_CSV)
    stops_csv = resolve_csv(args.stops_csv, data_dir, DEFAULT_STOPS_CSV)
    replay_csv = resolve_csv(args.replay_csv, data_dir, DEFAULT_ENTANGLEMENT_CSV)

    walking = read_samples(walking_csv)
    stops = read_samples(stops_csv)
    calibration = calibrate(walking, stops, args.rear_sigma, args.stop_threshold, args.stop_max_spike_fraction)
    if not args.summary_only:
        print_calibration(calibration, walking_csv, stops_csv)
        print(f"Replay CSV       : {replay_csv}")
        print()

    replay(
        samples=read_samples(replay_csv),
        calibration=calibration,
        window_seconds=args.window_seconds,
        persistence_seconds=args.persistence_seconds,
        required_fraction=args.required_window_fraction,
        cooldown_seconds=args.cooldown_seconds,
        summary_only=args.summary_only,
        log_normal=args.log_normal,
    )


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""Replay-only back-left entanglement alarm test script.

The script calibrates itself from the local CSVs:
  - walking_v2: normal rear thigh-torque trend
  - back_left_leg: labeled RL/back-left entanglement, default t=6s..10s
  - stops: front-left stop spikes used as a veto

It prints entanglement alarms while replaying a CSV. No ROS2 or pandas is
required; it reads the normalized lowstate CSV columns directly.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Deque, Iterable


TARGET_LEG = "RL"
REAR_LEGS = ("RR", "RL")
LEG_NAMES = {
    "FL": "front_left",
    "FR": "front_right",
    "RL": "back_left",
    "RR": "back_right",
}
JOINTS = ("hip", "thigh", "calf")

DEFAULT_WALKING_CSV = "go2_lowstate_walking_v2.csv"
DEFAULT_ENTANGLEMENT_CSV = "go2_lowstate_back_left_leg.csv"
DEFAULT_STOPS_CSV = "go2_lowstate_stops.csv"


@dataclass(frozen=True)
class Sample:
    timestamp: float
    values: dict[str, float]


@dataclass(frozen=True)
class Calibration:
    polarity: str
    thigh_threshold: float
    thigh_scale: float
    stop_fl_threshold: float
    stop_fl_scale: float
    normal_guard: float
    entanglement_tail: float
    stop_tail: float
    feature_names: tuple[str, ...]
    positive_center: dict[str, float]
    negative_center: dict[str, float]
    feature_scale: dict[str, float]
    event_threshold: float


@dataclass(frozen=True)
class WindowResult:
    timestamp: float
    score: float
    raw_score: float
    fraction: float
    duration: float
    level_score: float
    extreme_tau: float
    fl_extreme_tau: float
    rr_score: float
    stop_score: float
    stop_vetoed: bool
    stop_event: bool
    alarm: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a Go2 lowstate CSV and print back-left entanglement alarms."
    )
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=None,
        help="CSV to replay. Defaults to data/go2_lowstate_back_left_leg.csv.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Folder containing the Go2 CSV files.")
    parser.add_argument("--walking-csv", type=Path, default=None, help="Normal walking baseline CSV.")
    parser.add_argument("--entanglement-csv", type=Path, default=None, help="Back-left entanglement calibration CSV.")
    parser.add_argument("--stops-csv", type=Path, default=None, help="Random-stop calibration CSV.")
    parser.add_argument("--entanglement-start", type=float, default=6.0, help="Labeled entanglement start time in seconds.")
    parser.add_argument("--entanglement-end", type=float, default=10.0, help="Labeled entanglement end time in seconds.")
    parser.add_argument("--window-seconds", type=float, default=0.35, help="Rolling window length.")
    parser.add_argument("--persistence-seconds", type=float, default=0.22, help="Continuous threshold duration needed for alarm.")
    parser.add_argument("--required-window-fraction", type=float, default=0.58, help="Window fraction that must exceed threshold.")
    parser.add_argument("--cooldown-seconds", type=float, default=0.80, help="Minimum time between printed alarms.")
    parser.add_argument("--stop-veto-score", type=float, default=1.0, help="Front-left downward spike score that marks a stop.")
    parser.add_argument("--summary-only", action="store_true", help="Only print calibration and final replay summary.")
    parser.add_argument("--log-normal", action="store_true", help="Print periodic non-alarm status lines.")
    return parser.parse_args()


def safe_float(raw: str | None, default: float = 0.0) -> float:
    try:
        value = float(raw) if raw not in (None, "") else default
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def percentile(values: Iterable[float], q: float) -> float:
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


def median(values: Iterable[float]) -> float:
    return percentile(values, 50.0)


def robust_scale(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return 1.0
    center = median(clean)
    deviations = [abs(value - center) for value in clean]
    mad_sigma = 1.4826 * median(deviations)
    if mad_sigma > 1e-6:
        return mad_sigma
    if len(clean) > 1:
        mean_value = fmean(clean)
        variance = fmean((value - mean_value) ** 2 for value in clean)
        return max(math.sqrt(variance), 1e-6)
    return 1.0


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def find_data_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    base = script_dir()
    candidates = (
        base / "data",
        base / "data_analysis" / "data",
        base.parent / "data",
        Path.cwd() / "data",
    )
    for candidate in candidates:
        if (candidate / DEFAULT_WALKING_CSV).exists():
            return candidate
    return candidates[0]


def resolve_csv(path: Path | None, data_dir: Path, default_name: str) -> Path:
    if path is None:
        return data_dir / default_name
    if path.exists() or path.is_absolute():
        return path
    data_path = data_dir / path
    if data_path.exists():
        return data_path
    named_data_path = data_dir / path.name
    if named_data_path.exists():
        return named_data_path
    return path


def require_columns(fieldnames: list[str], columns: Iterable[str], csv_path: Path) -> None:
    missing = [column for column in columns if column not in fieldnames]
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")


def read_samples(csv_path: Path) -> list[Sample]:
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    required = ["timestamp", "FL_thigh_tau"]
    for leg in REAR_LEGS:
        required.extend([f"{leg}_thigh_tau", f"{leg}_thigh_dq"])
        required.extend(f"{leg}_{joint}_tau" for joint in JOINTS)
        required.extend(f"{leg}_{joint}_dq" for joint in JOINTS)

    samples: list[Sample] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")
        require_columns(reader.fieldnames, required, csv_path)

        first_timestamp: float | None = None
        for row in reader:
            raw_timestamp = safe_float(row.get("timestamp"))
            if first_timestamp is None:
                first_timestamp = raw_timestamp
            timestamp = raw_timestamp - first_timestamp
            values = {"FL_thigh_tau": safe_float(row.get("FL_thigh_tau"))}
            for leg in REAR_LEGS:
                values[f"{leg}_thigh_tau"] = safe_float(row.get(f"{leg}_thigh_tau"))
                values[f"{leg}_thigh_dq"] = safe_float(row.get(f"{leg}_thigh_dq"))
                values[f"{leg}_torque_abs_sum"] = sum(abs(safe_float(row.get(f"{leg}_{joint}_tau"))) for joint in JOINTS)
                values[f"{leg}_dq_abs_mean"] = fmean(abs(safe_float(row.get(f"{leg}_{joint}_dq"))) for joint in JOINTS)
            samples.append(Sample(timestamp=timestamp, values=values))

    if not samples:
        raise SystemExit(f"CSV contains no data rows: {csv_path}")
    return samples


def values_for(samples: Iterable[Sample], column: str, start: float | None = None, end: float | None = None) -> list[float]:
    result: list[float] = []
    for sample in samples:
        if start is not None and sample.timestamp < start:
            continue
        if end is not None and sample.timestamp > end:
            continue
        result.append(sample.values[column])
    return result


def choose_polarity(walking: list[Sample], entanglement: list[Sample], start: float, end: float) -> str:
    ent_values = values_for(entanglement, "RL_thigh_tau", start, end)
    normal_values = values_for(walking, "RL_thigh_tau")
    if not normal_values or not ent_values:
        return "down"
    normal_center = median(normal_values)
    entanglement_center = median(ent_values)
    return "up" if entanglement_center > normal_center else "down"


def calibrate(
    walking: list[Sample],
    stops: list[Sample],
    entanglement: list[Sample],
    start: float,
    end: float,
) -> Calibration:
    polarity = choose_polarity(walking, entanglement, start, end)
    normal_tau = values_for(walking, "RL_thigh_tau")
    ent_tau = values_for(entanglement, "RL_thigh_tau", start, end)
    if not ent_tau:
        raise SystemExit("No RL samples found in the labeled entanglement time window.")

    normal_scale = robust_scale(normal_tau)
    if polarity == "down":
        normal_guard = percentile(normal_tau, 25.0)
        ent_tail = percentile(ent_tau, 75.0)
        if ent_tail < normal_guard:
            thigh_threshold = (normal_guard + ent_tail) / 2.0
        else:
            thigh_threshold = percentile(ent_tau, 90.0)
        thigh_scale = max(abs(normal_guard - thigh_threshold), normal_scale, 1e-6)
    else:
        normal_guard = percentile(normal_tau, 75.0)
        ent_tail = percentile(ent_tau, 25.0)
        if ent_tail > normal_guard:
            thigh_threshold = (normal_guard + ent_tail) / 2.0
        else:
            thigh_threshold = percentile(ent_tau, 10.0)
        thigh_scale = max(abs(thigh_threshold - normal_guard), normal_scale, 1e-6)

    fl_normal = values_for(walking, "FL_thigh_tau") + values_for(entanglement, "FL_thigh_tau", start, end)
    fl_stops = values_for(stops, "FL_thigh_tau")
    fl_normal_guard = percentile(fl_normal, 0.5)
    stop_tail = percentile(fl_stops, 1.0)
    if stop_tail < fl_normal_guard:
        stop_fl_threshold = (stop_tail + fl_normal_guard) / 2.0
    else:
        stop_fl_threshold = fl_normal_guard - 3.0 * robust_scale(fl_normal)
    stop_fl_scale = max(abs(fl_normal_guard - stop_fl_threshold), robust_scale(fl_normal), 1e-6)

    return Calibration(
        polarity=polarity,
        thigh_threshold=thigh_threshold,
        thigh_scale=thigh_scale,
        stop_fl_threshold=stop_fl_threshold,
        stop_fl_scale=stop_fl_scale,
        normal_guard=normal_guard,
        entanglement_tail=ent_tail,
        stop_tail=stop_tail,
        feature_names=(),
        positive_center={},
        negative_center={},
        feature_scale={},
        event_threshold=1.0,
    )


def window_feature_values(samples: list[Sample]) -> dict[str, float]:
    rl_tau = [sample.values["RL_thigh_tau"] for sample in samples]
    rr_tau = [sample.values["RR_thigh_tau"] for sample in samples]
    rl_effort = [sample.values["RL_torque_abs_sum"] for sample in samples]
    rr_effort = [sample.values["RR_torque_abs_sum"] for sample in samples]
    rl_dq = [sample.values["RL_dq_abs_mean"] for sample in samples]
    rr_dq = [sample.values["RR_dq_abs_mean"] for sample in samples]
    fl_tau = [sample.values["FL_thigh_tau"] for sample in samples]
    return {
        "rl_mean": fmean(rl_tau),
        "rl_min": min(rl_tau),
        "rl_max": max(rl_tau),
        "rl_abs_mean": fmean(abs(value) for value in rl_tau),
        "rl_range": max(rl_tau) - min(rl_tau),
        "rl_effort_mean": fmean(rl_effort),
        "rl_effort_max": max(rl_effort),
        "rl_dq_mean": fmean(rl_dq),
        "rr_abs_mean": fmean(abs(value) for value in rr_tau),
        "rr_effort_mean": fmean(rr_effort),
        "rr_dq_mean": fmean(rr_dq),
        "rear_effort_gap": fmean(rl_effort) - fmean(rr_effort),
        "rear_abs_tau_gap": fmean(abs(value) for value in rl_tau) - fmean(abs(value) for value in rr_tau),
        "fl_min": min(fl_tau),
    }


def iter_window_features(samples: list[Sample], window_seconds: float) -> Iterable[tuple[float, dict[str, float]]]:
    history: Deque[Sample] = deque()
    for sample in samples:
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) >= 2:
            yield sample.timestamp, window_feature_values(list(history))


def center_for(feature_rows: list[dict[str, float]], feature_names: tuple[str, ...]) -> dict[str, float]:
    return {name: median(row[name] for row in feature_rows) for name in feature_names}


def distance_to_center(features: dict[str, float], center: dict[str, float], scale: dict[str, float], feature_names: tuple[str, ...]) -> float:
    if not feature_names:
        return 0.0
    normalized = [abs(features[name] - center[name]) / max(scale[name], 1e-6) for name in feature_names]
    return fmean(normalized)


def profile_score(features: dict[str, float], calibration: Calibration) -> float:
    positive_distance = distance_to_center(features, calibration.positive_center, calibration.feature_scale, calibration.feature_names)
    negative_distance = distance_to_center(features, calibration.negative_center, calibration.feature_scale, calibration.feature_names)
    return negative_distance / (positive_distance + 1e-6)


def tune_event_profile(
    calibration: Calibration,
    walking: list[Sample],
    stops: list[Sample],
    entanglement: list[Sample],
    window_seconds: float,
    start: float,
    end: float,
) -> Calibration:
    feature_names = (
        "rl_mean",
        "rl_min",
        "rl_max",
        "rl_abs_mean",
        "rl_range",
        "rl_effort_mean",
        "rl_effort_max",
        "rl_dq_mean",
        "rr_abs_mean",
        "rr_effort_mean",
        "rr_dq_mean",
        "rear_effort_gap",
        "rear_abs_tau_gap",
        "fl_min",
    )
    positive_rows: list[dict[str, float]] = []
    negative_rows: list[dict[str, float]] = []

    for timestamp, features in iter_window_features(entanglement, window_seconds):
        if start <= timestamp <= end:
            positive_rows.append(features)
        else:
            negative_rows.append(features)
    negative_rows.extend(features for _, features in iter_window_features(walking, window_seconds))
    negative_rows.extend(features for _, features in iter_window_features(stops, window_seconds))

    if not positive_rows or not negative_rows:
        return calibration

    positive_center = center_for(positive_rows, feature_names)
    negative_center = center_for(negative_rows, feature_names)
    all_rows = positive_rows + negative_rows
    feature_scale = {name: robust_scale(row[name] for row in all_rows) for name in feature_names}

    positive_scores = [
        distance_to_center(row, negative_center, feature_scale, feature_names)
        / (distance_to_center(row, positive_center, feature_scale, feature_names) + 1e-6)
        for row in positive_rows
    ]
    negative_scores = [
        distance_to_center(row, negative_center, feature_scale, feature_names)
        / (distance_to_center(row, positive_center, feature_scale, feature_names) + 1e-6)
        for row in negative_rows
    ]
    positive_guard = percentile(positive_scores, 15.0)
    negative_guard = max(negative_scores)
    if positive_guard > negative_guard:
        event_threshold = (positive_guard + negative_guard) / 2.0
    else:
        event_threshold = negative_guard * 1.01

    return Calibration(
        polarity=calibration.polarity,
        thigh_threshold=calibration.thigh_threshold,
        thigh_scale=calibration.thigh_scale,
        stop_fl_threshold=calibration.stop_fl_threshold,
        stop_fl_scale=calibration.stop_fl_scale,
        normal_guard=calibration.normal_guard,
        entanglement_tail=calibration.entanglement_tail,
        stop_tail=calibration.stop_tail,
        feature_names=feature_names,
        positive_center=positive_center,
        negative_center=negative_center,
        feature_scale=feature_scale,
        event_threshold=event_threshold,
    )


def is_exceedance(value: float, calibration: Calibration) -> bool:
    if calibration.polarity == "down":
        return value <= calibration.thigh_threshold
    return value >= calibration.thigh_threshold


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def alarm_header() -> str:
    return (
        f"{'time(s)':>8}  {'leg':<11}  {'score':>6}  {'tau':>9}  "
        f"{'frac':>5}  {'dur(s)':>6}  {'RR ctx':>6}  {'FL stop':>7}"
    )


def format_alarm_row(result: WindowResult) -> str:
    return (
        f"{result.timestamp:8.3f}  {TARGET_LEG + '/' + LEG_NAMES[TARGET_LEG]:<11}  "
        f"{result.score:6.2f}  {result.extreme_tau:9.3f}  {result.fraction:5.2f}  "
        f"{result.duration:6.3f}  {result.rr_score:6.2f}  {result.stop_score:7.2f}"
    )


def format_entangle_event(result: WindowResult) -> str:
    return (
        f"t={result.timestamp:.3f}s type=ENTANGLE leg={TARGET_LEG}/{LEG_NAMES[TARGET_LEG]} "
        f"score={result.score:.2f} tau={result.extreme_tau:.3f} "
        f"raw={result.raw_score:.2f} frac={result.fraction:.2f} dur={result.duration:.3f}s rr_ctx={result.rr_score:.2f}"
    )


def format_stop_event(result: WindowResult) -> str:
    return (
        f"t={result.timestamp:.3f}s type=STOP leg=FL/{LEG_NAMES['FL']} "
        f"score={result.stop_score:.2f} tau={result.fl_extreme_tau:.3f} "
        f"entangle_score={result.score:.2f}"
    )


def leg_score(samples: list[Sample], leg: str, calibration: Calibration, required_fraction: float, persistence_seconds: float) -> tuple[float, float, float, float, float]:
    if len(samples) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    tau_values = [sample.values[f"{leg}_thigh_tau"] for sample in samples]
    if calibration.polarity == "down":
        extreme_tau = min(tau_values)
        level_margin = calibration.thigh_threshold - extreme_tau
    else:
        extreme_tau = max(tau_values)
        level_margin = extreme_tau - calibration.thigh_threshold

    exceedance_flags = [is_exceedance(sample.values[f"{leg}_thigh_tau"], calibration) for sample in samples]
    fraction = sum(1 for flag in exceedance_flags if flag) / len(exceedance_flags)

    duration = 0.0
    if exceedance_flags[-1]:
        start_time = samples[-1].timestamp
        for sample, flag in zip(reversed(samples), reversed(exceedance_flags)):
            if not flag:
                break
            start_time = sample.timestamp
        duration = samples[-1].timestamp - start_time

    level_score = 1.0 + max(0.0, level_margin) / calibration.thigh_scale
    fraction_score = fraction / max(required_fraction, 1e-6)
    duration_score = duration / max(persistence_seconds, 1e-6)
    score = min(level_score, fraction_score, duration_score)
    return score, fraction, duration, level_score, extreme_tau


def evaluate_window(
    history: Deque[Sample],
    calibration: Calibration,
    required_fraction: float,
    persistence_seconds: float,
    stop_veto_score: float,
) -> WindowResult:
    samples = list(history)
    timestamp = samples[-1].timestamp
    raw_score, fraction, duration, level_score, extreme_tau = leg_score(
        samples, TARGET_LEG, calibration, required_fraction, persistence_seconds
    )
    rr_score, _, _, _, _ = leg_score(samples, "RR", calibration, required_fraction, persistence_seconds)
    event_score = profile_score(window_feature_values(samples), calibration)

    fl_min = min(sample.values["FL_thigh_tau"] for sample in samples)
    stop_score = max(0.0, (calibration.stop_fl_threshold - fl_min) / calibration.stop_fl_scale)
    stop_event = stop_score >= stop_veto_score
    stop_vetoed = stop_event and event_score < calibration.event_threshold * 1.25
    alarm = event_score >= calibration.event_threshold and raw_score >= 0.75 and not stop_vetoed

    return WindowResult(
        timestamp=timestamp,
        score=event_score,
        raw_score=raw_score,
        fraction=fraction,
        duration=duration,
        level_score=level_score,
        extreme_tau=extreme_tau,
        fl_extreme_tau=fl_min,
        rr_score=rr_score,
        stop_score=stop_score,
        stop_vetoed=stop_vetoed,
        stop_event=stop_event,
        alarm=alarm,
    )


def replay(
    samples: list[Sample],
    calibration: Calibration,
    window_seconds: float,
    persistence_seconds: float,
    required_fraction: float,
    cooldown_seconds: float,
    stop_veto_score: float,
    summary_only: bool,
    log_normal: bool,
) -> None:
    history: Deque[Sample] = deque()
    last_alarm_time = -1e9
    last_stop_time = -1e9
    alarm_count = 0
    stop_count = 0
    first_alarm: WindowResult | None = None
    first_stop: WindowResult | None = None
    best_result: WindowResult | None = None
    veto_count = 0

    if not summary_only:
        print_section("Replay")
        print(f"Window              : {window_seconds:.2f}s")
        print(f"Required persistence: {persistence_seconds:.2f}s")
        print(f"Required fraction   : {required_fraction:.2f}")
        print(f"Cooldown            : {cooldown_seconds:.2f}s")
        print(f"Stop veto score     : {stop_veto_score:.2f}")

    for row_index, sample in enumerate(samples):
        history.append(sample)
        cutoff = sample.timestamp - window_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()
        if len(history) < 2:
            continue

        result = evaluate_window(history, calibration, required_fraction, persistence_seconds, stop_veto_score)
        if best_result is None or result.score > best_result.score:
            best_result = result
        if result.stop_vetoed:
            veto_count += 1

        if result.stop_event and result.timestamp - last_stop_time >= cooldown_seconds:
            last_stop_time = result.timestamp
            stop_count += 1
            if first_stop is None:
                first_stop = result
            print(format_stop_event(result))

        if result.alarm and result.timestamp - last_alarm_time >= cooldown_seconds:
            last_alarm_time = result.timestamp
            alarm_count += 1
            if first_alarm is None:
                first_alarm = result
            print(format_entangle_event(result))
        elif log_normal and not summary_only and row_index % 50 == 0:
            state = "STOP_VETO" if result.stop_vetoed else "normal"
            print(
                f"status t={result.timestamp:8.3f}s  {state:<9}  "
                f"RL_score={result.score:5.2f}  RR_ctx={result.rr_score:5.2f}  FL_stop={result.stop_score:5.2f}"
            )

    print_section("Summary")
    print(f"Rows replayed       : {len(samples)}")
    print(f"Entangle events     : {alarm_count}")
    print(f"Stop events         : {stop_count}")
    print(f"Stop-veto windows   : {veto_count}")
    if first_alarm is None:
        print("First alarm         : none")
    else:
        print(
            f"First alarm         : t={first_alarm.timestamp:.3f}s  leg={TARGET_LEG}/{LEG_NAMES[TARGET_LEG]}  "
            f"score={first_alarm.score:.2f}  tau={first_alarm.extreme_tau:.3f}"
        )
    if first_stop is None:
        print("First stop          : none")
    else:
        print(
            f"First stop          : t={first_stop.timestamp:.3f}s  leg=FL/{LEG_NAMES['FL']}  "
            f"score={first_stop.stop_score:.2f}  tau={first_stop.fl_extreme_tau:.3f}"
        )
    if best_result is not None:
        print(
            f"Max RL score        : t={best_result.timestamp:.3f}s  score={best_result.score:.2f}  "
            f"tau={best_result.extreme_tau:.3f}  RR_ctx={best_result.rr_score:.2f}"
        )


def print_calibration(calibration: Calibration, walking_csv: Path, entanglement_csv: Path, stops_csv: Path) -> None:
    print_section("Calibration")
    print(f"Walking baseline    : {format_path(walking_csv)}")
    print(f"Entanglement label  : {format_path(entanglement_csv)}")
    print(f"Stop label          : {format_path(stops_csv)}")
    print()
    print("Back-left rule")
    print(
        f"  leg                : {TARGET_LEG}/{LEG_NAMES[TARGET_LEG]}\n"
        f"  direction          : thigh_tau {calibration.polarity}\n"
        f"  alarm threshold    : {calibration.thigh_threshold:.3f}\n"
        f"  event score gate   : {calibration.event_threshold:.3f}\n"
        f"  normal guard       : {calibration.normal_guard:.3f}\n"
        f"  entanglement tail  : {calibration.entanglement_tail:.3f}"
    )
    print()
    print("Stop veto")
    print(
        f"  leg                : FL/{LEG_NAMES['FL']}\n"
        f"  downward threshold : {calibration.stop_fl_threshold:.3f}\n"
        f"  stops tail         : {calibration.stop_tail:.3f}"
    )


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    walking_csv = resolve_csv(args.walking_csv, data_dir, DEFAULT_WALKING_CSV)
    entanglement_csv = resolve_csv(args.entanglement_csv, data_dir, DEFAULT_ENTANGLEMENT_CSV)
    stops_csv = resolve_csv(args.stops_csv, data_dir, DEFAULT_STOPS_CSV)
    replay_csv = resolve_csv(args.replay_csv, data_dir, DEFAULT_ENTANGLEMENT_CSV)

    walking = read_samples(walking_csv)
    entanglement = read_samples(entanglement_csv)
    stops = read_samples(stops_csv)
    calibration = calibrate(walking, stops, entanglement, args.entanglement_start, args.entanglement_end)
    calibration = tune_event_profile(
        calibration,
        walking,
        stops,
        entanglement,
        args.window_seconds,
        args.entanglement_start,
        args.entanglement_end,
    )
    if not args.summary_only:
        print_calibration(calibration, walking_csv, entanglement_csv, stops_csv)

    if not args.summary_only:
        print()
        print(f"Replay CSV          : {format_path(replay_csv)}")
    replay_samples = read_samples(replay_csv)
    replay(
        samples=replay_samples,
        calibration=calibration,
        window_seconds=args.window_seconds,
        persistence_seconds=args.persistence_seconds,
        required_fraction=args.required_window_fraction,
        cooldown_seconds=args.cooldown_seconds,
        stop_veto_score=args.stop_veto_score,
        summary_only=args.summary_only,
        log_normal=args.log_normal,
    )


if __name__ == "__main__":
    main()
'''