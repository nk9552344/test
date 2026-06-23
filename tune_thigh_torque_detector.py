#!/usr/bin/env python3
"""One-time hyperparameter sweep for the thigh-torque entanglement detector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from thigh_torque_entanglement_detector import (
    DEFAULT_MODEL_CONFIG_PATH,
    LEG_ORDER,
    ThighTorqueDetector,
    calibrate_models,
    parse_entanglement_cases,
    read_csv_samples,
    save_model_config,
)


NEGATIVE_CSVS = (
    Path("data/go2_lowstate_walking_v2.csv"),
    Path("data/go2_lowstate_walking.csv"),
)

STRONG_POSITIVE_CASES = (
    (Path("data/go2_lowstate_1781770500.csv"), ("RL",)),
    (Path("data/go2_lowstate_1781770653.csv"), ("RR",)),
    (Path("data/go2_lowstate_back_both_leg.csv"), ("RR", "RL")),
    (Path("data/go2_lowstate_front_both_leg.csv"), ("FR", "FL")),
)

WEAK_VALIDATION_CASES = (
    (Path("data/go2_lowstate_front_left.csv"), ("FL",)),
    (Path("data/go2_lowstate_front_right.csv"), ("FR",)),
)


@dataclass(frozen=True)
class ReplaySummary:
    alarm_events: int
    first_alarm_time: float | None
    first_alarm_leg: str | None
    expected_hit: bool
    wrong_first_leg: bool
    max_score: float
    max_leg: str | None
    max_time: float | None


@dataclass(frozen=True)
class Candidate:
    sort_key: tuple[float, ...]
    polarity_mode: str
    walking_lower_percentile: float
    entanglement_lower_percentile: float
    threshold_blend: float
    window_seconds: float
    persistence_seconds: float
    required_window_fraction: float
    negative_results: list[tuple[Path, ReplaySummary]]
    strong_results: list[tuple[Path, tuple[str, ...], ReplaySummary]]
    weak_results: list[tuple[Path, tuple[str, ...], ReplaySummary]]


def existing_cases(cases: tuple[tuple[Path, tuple[str, ...]], ...]) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    return tuple((csv_path, legs) for csv_path, legs in cases if csv_path.exists())


def polarity_map(mode: str) -> dict[str, str] | None:
    if mode == "signed":
        return None
    if mode == "all_down":
        return {leg: "down" for leg in LEG_ORDER}
    if mode == "all_up":
        return {leg: "up" for leg in LEG_ORDER}
    raise ValueError(f"Unknown polarity mode: {mode}")


def replay(csv_path: Path, detector: ThighTorqueDetector, expected_legs: tuple[str, ...] | None) -> ReplaySummary:
    samples_by_leg, _ = read_csv_samples(csv_path)
    row_count = max(len(samples) for samples in samples_by_leg.values())

    alarm_events = 0
    first_alarm_time: float | None = None
    first_alarm_leg: str | None = None
    expected_hit = False
    wrong_first_leg = False
    max_score = 0.0
    max_leg: str | None = None
    max_time: float | None = None
    last_alarm_leg: str | None = None

    for row_index in range(row_count):
        sample_group = {leg: samples[row_index] for leg, samples in samples_by_leg.items() if row_index < len(samples)}
        if not sample_group:
            continue
        timestamp = max(sample.timestamp for sample in sample_group.values())
        result = detector.add_samples(sample_group)

        if result.candidate_score > max_score:
            max_score = result.candidate_score
            max_leg = result.candidate_leg
            max_time = timestamp

        if result.alarm_leg is None:
            last_alarm_leg = None
            continue
        if result.alarm_leg == last_alarm_leg:
            continue

        alarm_events += 1
        last_alarm_leg = result.alarm_leg
        if first_alarm_time is None:
            first_alarm_time = timestamp
            first_alarm_leg = result.alarm_leg
            wrong_first_leg = expected_legs is not None and result.alarm_leg not in expected_legs
        if expected_legs is not None and result.alarm_leg in expected_legs:
            expected_hit = True

    return ReplaySummary(
        alarm_events=alarm_events,
        first_alarm_time=first_alarm_time,
        first_alarm_leg=first_alarm_leg,
        expected_hit=expected_hit,
        wrong_first_leg=wrong_first_leg,
        max_score=max_score,
        max_leg=max_leg,
        max_time=max_time,
    )


def evaluate_candidate(args: argparse.Namespace, polarity_mode: str, walking_lower_percentile: float, entanglement_lower_percentile: float, threshold_blend: float, window_seconds: float, persistence_seconds: float, required_window_fraction: float) -> Candidate | None:
    training_cases = parse_entanglement_cases(args.entanglement_case)
    models = calibrate_models(
        walking_csv=args.walking_csv,
        entanglement_cases=training_cases,
        calibration_seconds=args.calibration_seconds,
        walking_lower_percentile=walking_lower_percentile,
        entanglement_lower_percentile=entanglement_lower_percentile,
        threshold_blend=threshold_blend,
        polarity_by_leg=polarity_map(polarity_mode),
    )

    negative_results: list[tuple[Path, ReplaySummary]] = []
    for csv_path in NEGATIVE_CSVS:
        if csv_path.exists():
            detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
            negative_results.append((csv_path, replay(csv_path, detector, None)))

    strong_results: list[tuple[Path, tuple[str, ...], ReplaySummary]] = []
    for csv_path, expected_legs in existing_cases(STRONG_POSITIVE_CASES):
        detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
        strong_results.append((csv_path, expected_legs, replay(csv_path, detector, expected_legs)))

    weak_results: list[tuple[Path, tuple[str, ...], ReplaySummary]] = []
    for csv_path, expected_legs in existing_cases(WEAK_VALIDATION_CASES):
        detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
        weak_results.append((csv_path, expected_legs, replay(csv_path, detector, expected_legs)))

    if not strong_results:
        return None

    false_alarm_events = sum(summary.alarm_events for _, summary in negative_results)
    strong_misses = sum(1 for _, _, summary in strong_results if not summary.expected_hit)
    wrong_first = sum(1 for _, _, summary in strong_results if summary.wrong_first_leg)
    weak_misses = sum(1 for _, _, summary in weak_results if not summary.expected_hit)
    first_alarm_times = [summary.first_alarm_time for _, _, summary in strong_results if summary.first_alarm_time is not None]
    mean_first_alarm_time = sum(first_alarm_times) / len(first_alarm_times) if first_alarm_times else 1e9
    total_positive_events = sum(summary.alarm_events for _, _, summary in strong_results)
    polarity_penalty = 0 if polarity_mode == "signed" else 1

    sort_key = (
        float(false_alarm_events),
        float(strong_misses),
        float(wrong_first),
        float(weak_misses) * 0.25,
        mean_first_alarm_time,
        float(total_positive_events),
        float(polarity_penalty),
    )

    return Candidate(
        sort_key=sort_key,
        polarity_mode=polarity_mode,
        walking_lower_percentile=walking_lower_percentile,
        entanglement_lower_percentile=entanglement_lower_percentile,
        threshold_blend=threshold_blend,
        window_seconds=window_seconds,
        persistence_seconds=persistence_seconds,
        required_window_fraction=required_window_fraction,
        negative_results=negative_results,
        strong_results=strong_results,
        weak_results=weak_results,
    )


def print_summary(candidate: Candidate) -> None:
    print("Best thigh-torque detector parameters:")
    print(f"  polarity_mode={candidate.polarity_mode}")
    print(f"  walking_lower_percentile={candidate.walking_lower_percentile}")
    print(f"  entanglement_lower_percentile={candidate.entanglement_lower_percentile}")
    print(f"  threshold_blend={candidate.threshold_blend}")
    print(f"  window_seconds={candidate.window_seconds}")
    print(f"  persistence_seconds={candidate.persistence_seconds}")
    print(f"  required_window_fraction={candidate.required_window_fraction}")

    print("Negative replay:")
    for csv_path, summary in candidate.negative_results:
        max_time = "n/a" if summary.max_time is None else f"{summary.max_time:.3f}s"
        print(f"  {csv_path.name}: alarms={summary.alarm_events}, max={summary.max_score:.2f} {summary.max_leg} at {max_time}")

    print("Strong positive replay:")
    for csv_path, expected_legs, summary in candidate.strong_results:
        first_time = "n/a" if summary.first_alarm_time is None else f"{summary.first_alarm_time:.3f}s"
        print(f"  {csv_path.name}: expected={','.join(expected_legs)} alarms={summary.alarm_events}, first={summary.first_alarm_leg} at {first_time}, hit={summary.expected_hit}")

    if candidate.weak_results:
        print("Weak/anomalous validation replay:")
        for csv_path, expected_legs, summary in candidate.weak_results:
            first_time = "n/a" if summary.first_alarm_time is None else f"{summary.first_alarm_time:.3f}s"
            print(f"  {csv_path.name}: expected={','.join(expected_legs)} alarms={summary.alarm_events}, first={summary.first_alarm_leg} at {first_time}, hit={summary.expected_hit}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep signed thigh-torque detector parameters and save the best fixed JSON config.")
    parser.add_argument("--walking-csv", type=Path, default=Path("data/go2_lowstate_walking_v2.csv"))
    parser.add_argument("--entanglement-case", action="append", default=None, metavar="CSV:LEG[,LEG]")
    parser.add_argument("--calibration-seconds", type=float, default=1e9)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--top", type=int, default=5, help="How many top candidates to print after the winner.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: list[Candidate] = []

    for polarity_mode in ("signed", "all_down", "all_up"):
        for walking_lower_percentile in (0.5, 1.0, 2.0):
            for entanglement_lower_percentile in (2.0, 5.0, 10.0):
                for threshold_blend in (0.25, 0.5, 0.75):
                    for window_seconds in (0.12, 0.20, 0.30):
                        for persistence_seconds in (0.06, 0.10, 0.14):
                            for required_window_fraction in (0.50, 0.65, 0.80):
                                candidate = evaluate_candidate(
                                    args,
                                    polarity_mode,
                                    walking_lower_percentile,
                                    entanglement_lower_percentile,
                                    threshold_blend,
                                    window_seconds,
                                    persistence_seconds,
                                    required_window_fraction,
                                )
                                if candidate is not None:
                                    candidates.append(candidate)

    if not candidates:
        raise SystemExit("No labeled positive CSVs were available for tuning.")

    candidates.sort(key=lambda candidate: candidate.sort_key)
    best = candidates[0]
    print_summary(best)

    print(f"Top {min(args.top, len(candidates))} candidates:")
    for rank, candidate in enumerate(candidates[: args.top], start=1):
        print(
            f"  {rank}. key={candidate.sort_key} polarity={candidate.polarity_mode} "
            f"wp={candidate.walking_lower_percentile} ep={candidate.entanglement_lower_percentile} "
            f"blend={candidate.threshold_blend} window={candidate.window_seconds} "
            f"persist={candidate.persistence_seconds} fraction={candidate.required_window_fraction}"
        )

    best_models = calibrate_models(
        walking_csv=args.walking_csv,
        entanglement_cases=parse_entanglement_cases(args.entanglement_case),
        calibration_seconds=args.calibration_seconds,
        walking_lower_percentile=best.walking_lower_percentile,
        entanglement_lower_percentile=best.entanglement_lower_percentile,
        threshold_blend=best.threshold_blend,
        polarity_by_leg=polarity_map(best.polarity_mode),
    )
    save_args = argparse.Namespace(
        walking_csv=args.walking_csv,
        calibration_seconds=args.calibration_seconds,
        window_seconds=best.window_seconds,
        persistence_seconds=best.persistence_seconds,
        required_window_fraction=best.required_window_fraction,
        cooldown_seconds=1.0,
        walking_lower_percentile=best.walking_lower_percentile,
        entanglement_lower_percentile=best.entanglement_lower_percentile,
        threshold_blend=best.threshold_blend,
    )
    save_model_config(args.model_config, best_models, save_args)
    print(f"Saved fixed detector parameters to {args.model_config}")


if __name__ == "__main__":
    main()