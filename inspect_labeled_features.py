#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

LEGS = ("FR", "FL", "RR", "RL")
JOINTS = ("hip", "thigh", "calf")
FILES = [
    ("walking", Path("data/go2_lowstate_walking.csv"), None),
    ("500", Path("data/go2_lowstate_1781770500.csv"), "RL"),
    ("653", Path("data/go2_lowstate_1781770653.csv"), "RR"),
]

def features(df, leg):
    tau_sum = df[[f"{leg}_{j}_tau" for j in JOINTS]].abs().sum(axis=1)
    dq_mean = df[[f"{leg}_{j}_dq" for j in JOINTS]].abs().mean(axis=1)
    q_mean = df[[f"{leg}_{j}_q" for j in JOINTS]].abs().mean(axis=1)
    foot = df[f"foot_{leg}"]
    return {
        "tau_sum": tau_sum,
        "dq_mean": dq_mean,
        "q_mean": q_mean,
        "foot": foot,
        "tau_per_dq": tau_sum / (dq_mean + 0.05),
        "foot_per_dq": foot / (dq_mean + 0.05),
    }

base = pd.read_csv("data/go2_lowstate_walking.csv")
base_stats = {leg: features(base, leg) for leg in LEGS}

for name, path, label_leg in FILES:
    df = pd.read_csv(path)
    ts = pd.to_numeric(df["timestamp"])
    elapsed = ts - ts.iloc[0]
    print(f"\n{name} {path.name} label={label_leg}")
    legs_to_show = [label_leg] if label_leg else list(LEGS)
    for leg in legs_to_show:
        fs = features(df, leg)
        print(f"  {leg}")
        for feature_name, series in fs.items():
            idx = int(series.idxmax())
            print(f"    {feature_name}: max={float(series.max()):.3f} at {float(elapsed.iloc[idx]):.3f}s mean={float(series.mean()):.3f}")
            b = base_stats[leg][feature_name]
            print(f"      walking same leg max={float(b.max()):.3f} mean={float(b.mean()):.3f}")
