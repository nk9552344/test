#!/usr/bin/env python3
"""One-time hyperparameter sweep for the thigh-torque entanglement detector."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
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
    (Path("data/go2_lowstate_back_left_leg.csv"), ("RL",)),
    (Path("data/go2_lowstate_back_right_leg.csv"), ("RR",)),
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
    expected_max_score: float
    expected_max_leg: str | None
    expected_max_time: float | None


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
    if "_" in mode:
        parsed: dict[str, str] = {}
        for chunk in mode.split("_"):
            leg = chunk[:2]
            polarity = chunk[2:]
            if leg not in LEG_ORDER or polarity not in ("up", "down"):
                raise ValueError(f"Invalid per-leg polarity chunk: {chunk!r}")
            parsed[leg] = polarity
        if set(parsed) != set(LEG_ORDER):
            raise ValueError(f"Per-leg polarity mode must include every leg: {mode}")
        return parsed
    raise ValueError(f"Unknown polarity mode: {mode}")


def per_leg_polarity_modes() -> tuple[str, ...]:
    modes: list[str] = []
    for fr in ("up", "down"):
        for fl in ("up", "down"):
            for rr in ("up", "down"):
                for rl in ("up", "down"):
                    modes.append(f"FR{fr}_FL{fl}_RR{rr}_RL{rl}")
    return tuple(modes)


@lru_cache(maxsize=None)
def cached_models(
    walking_csv: str,
    calibration_seconds: float,
    polarity_mode: str,
    walking_lower_percentile: float,
    entanglement_lower_percentile: float,
    threshold_blend: float,
    raw_entanglement_cases: tuple[str, ...] | None,
):
    return calibrate_models(
        walking_csv=Path(walking_csv),
        entanglement_cases=parse_entanglement_cases(list(raw_entanglement_cases) if raw_entanglement_cases else None),
        calibration_seconds=calibration_seconds,
        walking_lower_percentile=walking_lower_percentile,
        entanglement_lower_percentile=entanglement_lower_percentile,
        threshold_blend=threshold_blend,
        polarity_by_leg=polarity_map(polarity_mode),
    )


@lru_cache(maxsize=None)
def cached_samples(csv_path: str):
    return read_csv_samples(Path(csv_path))[0]


def replay(csv_path: Path, detector: ThighTorqueDetector, expected_legs: tuple[str, ...] | None, stride: int) -> ReplaySummary:
    samples_by_leg = cached_samples(str(csv_path))
    row_count = max(len(samples) for samples in samples_by_leg.values())

    alarm_events = 0
    first_alarm_time: float | None = None
    first_alarm_leg: str | None = None
    expected_hit = False
    wrong_first_leg = False
    max_score = 0.0
    max_leg: str | None = None
    max_time: float | None = None
    expected_max_score = 0.0
    expected_max_leg: str | None = None
    expected_max_time: float | None = None
    last_alarm_leg: str | None = None

    for row_index in range(0, row_count, max(1, stride)):
        sample_group = {leg: samples[row_index] for leg, samples in samples_by_leg.items() if row_index < len(samples)}
        if not sample_group:
            continue
        timestamp = max(sample.timestamp for sample in sample_group.values())
        result = detector.add_samples(sample_group)

        if result.candidate_score > max_score:
            max_score = result.candidate_score
            max_leg = result.candidate_leg
            max_time = timestamp

        if expected_legs is not None:
            for expected_leg in expected_legs:
                expected_score = result.scores.get(expected_leg, 0.0)
                if expected_score > expected_max_score:
                    expected_max_score = expected_score
                    expected_max_leg = expected_leg
                    expected_max_time = timestamp

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
        expected_max_score=expected_max_score,
        expected_max_leg=expected_max_leg,
        expected_max_time=expected_max_time,
    )


def evaluate_candidate(args: argparse.Namespace, polarity_mode: str, walking_lower_percentile: float, entanglement_lower_percentile: float, threshold_blend: float, window_seconds: float, persistence_seconds: float, required_window_fraction: float) -> Candidate | None:
    models = cached_models(
        str(args.walking_csv),
        args.calibration_seconds,
        polarity_mode,
        walking_lower_percentile,
        entanglement_lower_percentile,
        threshold_blend,
        tuple(args.entanglement_case) if args.entanglement_case else None,
    )

    negative_results: list[tuple[Path, ReplaySummary]] = []
    for csv_path in NEGATIVE_CSVS:
        if csv_path.exists():
            detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
            negative_results.append((csv_path, replay(csv_path, detector, None, args.stride)))

    if not args.allow_false_alarms and sum(summary.alarm_events for _, summary in negative_results) > 0:
        return None

    strong_results: list[tuple[Path, tuple[str, ...], ReplaySummary]] = []
    for csv_path, expected_legs in existing_cases(STRONG_POSITIVE_CASES):
        detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
        strong_results.append((csv_path, expected_legs, replay(csv_path, detector, expected_legs, args.stride)))

    strong_misses = sum(1 for _, _, summary in strong_results if not summary.expected_hit)
    wrong_first = sum(1 for _, _, summary in strong_results if summary.wrong_first_leg)
    expected_score_deficit = sum(max(0.0, 1.0 - summary.expected_max_score) for _, _, summary in strong_results)

    if not args.allow_strong_misses and any(not summary.expected_hit for _, _, summary in strong_results):
        return Candidate(
            sort_key=(
                float(sum(summary.alarm_events for _, summary in negative_results)),
                float(strong_misses),
                expected_score_deficit,
                float(wrong_first),
                0.0,
                1e9,
                float(sum(summary.alarm_events for _, _, summary in strong_results)),
                1.0,
            ),
            polarity_mode=polarity_mode,
            walking_lower_percentile=walking_lower_percentile,
            entanglement_lower_percentile=entanglement_lower_percentile,
            threshold_blend=threshold_blend,
            window_seconds=window_seconds,
            persistence_seconds=persistence_seconds,
            required_window_fraction=required_window_fraction,
            negative_results=negative_results,
            strong_results=strong_results,
            weak_results=[],
        )

    weak_results: list[tuple[Path, tuple[str, ...], ReplaySummary]] = []
    for csv_path, expected_legs in existing_cases(WEAK_VALIDATION_CASES):
        detector = ThighTorqueDetector(models, window_seconds, persistence_seconds, required_window_fraction)
        weak_results.append((csv_path, expected_legs, replay(csv_path, detector, expected_legs, args.stride)))

    if not strong_results:
        return None

    false_alarm_events = sum(summary.alarm_events for _, summary in negative_results)
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
        expected_max_time = "n/a" if summary.expected_max_time is None else f"{summary.expected_max_time:.3f}s"
        print(f"  {csv_path.name}: expected={','.join(expected_legs)} alarms={summary.alarm_events}, first={summary.first_alarm_leg} at {first_time}, hit={summary.expected_hit}, expected_max={summary.expected_max_score:.2f} {summary.expected_max_leg} at {expected_max_time}")

    if candidate.weak_results:
        print("Weak/anomalous validation replay:")
        for csv_path, expected_legs, summary in candidate.weak_results:
            first_time = "n/a" if summary.first_alarm_time is None else f"{summary.first_alarm_time:.3f}s"
            expected_max_time = "n/a" if summary.expected_max_time is None else f"{summary.expected_max_time:.3f}s"
            print(f"  {csv_path.name}: expected={','.join(expected_legs)} alarms={summary.alarm_events}, first={summary.first_alarm_leg} at {first_time}, hit={summary.expected_hit}, expected_max={summary.expected_max_score:.2f} {summary.expected_max_leg} at {expected_max_time}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep signed thigh-torque detector parameters and save the best fixed JSON config.")
    parser.add_argument("--walking-csv", type=Path, default=Path("data/go2_lowstate_walking_v2.csv"))
    parser.add_argument("--entanglement-case", action="append", default=None, metavar="CSV:LEG[,LEG]")
    parser.add_argument("--calibration-seconds", type=float, default=1e9)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--top", type=int, default=5, help="How many top candidates to print after the winner.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller first-pass grid.")
    parser.add_argument("--polarity-mode", choices=("signed", "all_down", "all_up", "per_leg"), default=None, help="Evaluate only one polarity mode, or per_leg to sweep all 16 leg-specific direction maps.")
    parser.add_argument("--allow-strong-misses", action="store_true", help="Allow saving a candidate that misses a strong labeled case.")
    parser.add_argument("--allow-false-alarms", action="store_true", help="Allow saving a candidate that alarms on walking logs.")
    parser.add_argument("--stride", type=int, default=1, help="Replay every Nth row for faster coarse sweeps. Use 1 for final validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: list[Candidate] = []

    if args.polarity_mode == "per_leg":
        polarity_modes = per_leg_polarity_modes()
    elif args.polarity_mode:
        polarity_modes = (args.polarity_mode,)
    else:
        polarity_modes = ("signed", "all_down", "all_up")
    walking_percentiles = (0.5, 1.0) if args.quick else (0.5, 1.0, 2.0)
    entanglement_percentiles = (5.0,) if args.quick else (2.0, 5.0, 10.0)
    threshold_blends = (0.25, 0.5, 0.75) if args.quick else (0.25, 0.5, 0.75)
    window_seconds_values = (0.12, 0.20, 0.30) if args.quick else (0.12, 0.20, 0.30)
    persistence_seconds_values = (0.04, 0.06, 0.10) if args.quick else (0.06, 0.10, 0.14)
    required_fraction_values = (0.35, 0.50, 0.65) if args.quick else (0.50, 0.65, 0.80)
    total = (
        len(polarity_modes)
        * len(walking_percentiles)
        * len(entanglement_percentiles)
        * len(threshold_blends)
        * len(window_seconds_values)
        * len(persistence_seconds_values)
        * len(required_fraction_values)
    )
    checked = 0
    print(f"Evaluating {total} candidates...")

    for polarity_mode in polarity_modes:
        for walking_lower_percentile in walking_percentiles:
            for entanglement_lower_percentile in entanglement_percentiles:
                for threshold_blend in threshold_blends:
                    for window_seconds in window_seconds_values:
                        for persistence_seconds in persistence_seconds_values:
                            for required_window_fraction in required_fraction_values:
                                checked += 1
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
                                if checked == 1 or checked % 25 == 0 or checked == total:
                                    print(f"Checked {checked}/{total} candidates", flush=True)

    if not candidates:
        raise SystemExit("No labeled positive CSVs were available for tuning.")

    candidates.sort(key=lambda candidate: candidate.sort_key)
    viable_candidates = candidates
    if not args.allow_strong_misses:
        viable_candidates = [
            candidate for candidate in viable_candidates
            if all(summary.expected_hit for _, _, summary in candidate.strong_results)
        ]
    if not args.allow_false_alarms:
        viable_candidates = [
            candidate for candidate in viable_candidates
            if sum(summary.alarm_events for _, summary in candidate.negative_results) == 0
        ]
    if not viable_candidates:
        print("No candidate passed the strict criteria: all strong cases detected and zero walking false alarms. Showing best imperfect candidate; not saving config.")
        print_summary(candidates[0])
        print(f"Top {min(args.top, len(candidates))} imperfect candidates:")
        for rank, candidate in enumerate(candidates[: args.top], start=1):
            print(
                f"  {rank}. key={candidate.sort_key} polarity={candidate.polarity_mode} "
                f"wp={candidate.walking_lower_percentile} ep={candidate.entanglement_lower_percentile} "
                f"blend={candidate.threshold_blend} window={candidate.window_seconds} "
                f"persist={candidate.persistence_seconds} fraction={candidate.required_window_fraction}"
            )
        raise SystemExit(2)

    candidates = viable_candidates
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
    saved_polarities = ", ".join(f"{leg}:{best_models[leg].polarity}" for leg in LEG_ORDER)
    print(f"Saved polarities: {saved_polarities}")
    print(f"Saved fixed detector parameters to {args.model_config}")


if __name__ == "__main__":
    main()