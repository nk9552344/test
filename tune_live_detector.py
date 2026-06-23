#!/usr/bin/env python3
"""Tune live entanglement detector parameters using labeled CSV logs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from live_entanglement_detector import (
    EntanglementEngine,
    calibrate_baseline,
    extract_csv_sample,
)

BASELINE_CSV = Path("data/go2_lowstate_walking.csv")
NEGATIVE_CSVS = [Path("data/go2_lowstate_walking.csv")]
POSITIVE_CSVS = [
    Path("data/go2_lowstate_1781770500.csv"),
    Path("data/go2_lowstate_1781770653.csv"),
]


@lru_cache(maxsize=None)
def cached_baseline(score_percentile: float):
    return calibrate_baseline(BASELINE_CSV, score_percentile)


@lru_cache(maxsize=None)
def cached_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    timestamps = pd.to_numeric(data["timestamp"], errors="coerce")
    elapsed = (timestamps - timestamps.iloc[0]).astype(float)
    return data, elapsed


@dataclass(frozen=True)
class ReplaySummary:
    alarm_count: int
    first_alarm_time: float | None
    first_alarm_leg: str | None
    max_candidate_score: float
    max_candidate_leg: str | None
    max_candidate_time: float | None


def replay(
    csv_path: Path,
    score_percentile: float,
    window_seconds: float,
    persistence_windows: int,
    dominance_gap: float,
    threshold_scale: float,
) -> ReplaySummary:
    baseline = cached_baseline(score_percentile)
    engine = EntanglementEngine(
        baseline=baseline,
        window_seconds=window_seconds,
        persistence_windows=persistence_windows,
        dominance_gap=dominance_gap,
        threshold_scale=threshold_scale,
    )

    data, elapsed = cached_csv(csv_path)

    alarm_count = 0
    first_alarm_time = None
    first_alarm_leg = None
    max_candidate_score = -1.0
    max_candidate_leg = None
    max_candidate_time = None

    for row_index, (_, row) in enumerate(data.iterrows()):
        result = engine.add_sample(float(elapsed.iloc[row_index]), extract_csv_sample(row))
        if result.candidate_score > max_candidate_score:
            max_candidate_score = result.candidate_score
            max_candidate_leg = result.candidate_leg
            max_candidate_time = float(elapsed.iloc[row_index])
        if result.alarm_leg is not None:
            alarm_count += 1
            if first_alarm_time is None:
                first_alarm_time = float(elapsed.iloc[row_index])
                first_alarm_leg = result.alarm_leg

    return ReplaySummary(
        alarm_count=alarm_count,
        first_alarm_time=first_alarm_time,
        first_alarm_leg=first_alarm_leg,
        max_candidate_score=max_candidate_score,
        max_candidate_leg=max_candidate_leg,
        max_candidate_time=max_candidate_time,
    )


def main() -> None:
    candidates = []
    for score_percentile in (97.5, 99.0, 99.5):
        for window_seconds in (0.20, 0.30, 0.40):
            for persistence_windows in (2, 3, 4):
                for dominance_gap in (0.0, 0.25, 0.5):
                    for threshold_scale in (0.45, 0.55, 0.65, 0.75, 0.85):
                        negatives = [replay(csv_path, score_percentile, window_seconds, persistence_windows, dominance_gap, threshold_scale) for csv_path in NEGATIVE_CSVS]
                        positives = [replay(csv_path, score_percentile, window_seconds, persistence_windows, dominance_gap, threshold_scale) for csv_path in POSITIVE_CSVS]
                        false_alarms = sum(item.alarm_count for item in negatives)
                        detected_positives = sum(1 for item in positives if item.alarm_count > 0)
                        total_positive_alarms = sum(item.alarm_count for item in positives)
                        if false_alarms == 0 and detected_positives == len(POSITIVE_CSVS):
                            candidates.append((
                                score_percentile,
                                window_seconds,
                                persistence_windows,
                                dominance_gap,
                                threshold_scale,
                                total_positive_alarms,
                                positives,
                                negatives,
                            ))

    if not candidates:
        print("No parameter set detected both positive logs while keeping walking at zero alarms.")
        return

    candidates.sort(key=lambda item: (item[5], -item[0], item[1], item[2], item[3], -item[4]))
    best = candidates[0]
    score_percentile, window_seconds, persistence_windows, dominance_gap, threshold_scale, total_positive_alarms, positives, negatives = best
    print("Best parameters:")
    print(f"  score_percentile={score_percentile}")
    print(f"  window_seconds={window_seconds}")
    print(f"  persistence_windows={persistence_windows}")
    print(f"  dominance_gap={dominance_gap}")
    print(f"  threshold_scale={threshold_scale}")
    print(f"  total_positive_alarms={total_positive_alarms}")
    print("Negative replay:")
    for csv_path, summary in zip(NEGATIVE_CSVS, negatives, strict=True):
        print(f"  {csv_path.name}: alarms={summary.alarm_count}, max={summary.max_candidate_score:.2f} {summary.max_candidate_leg} at {summary.max_candidate_time:.3f}s")
    print("Positive replay:")
    for csv_path, summary in zip(POSITIVE_CSVS, positives, strict=True):
        print(
            f"  {csv_path.name}: alarms={summary.alarm_count}, first={summary.first_alarm_leg} "
            f"at {summary.first_alarm_time:.3f}s, max={summary.max_candidate_score:.2f} "
            f"{summary.max_candidate_leg} at {summary.max_candidate_time:.3f}s"
        )


if __name__ == "__main__":
    main()
