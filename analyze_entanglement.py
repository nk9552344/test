#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

CSV = Path('data/go2_lowstate_1781770500.csv')
if not CSV.exists():
    raise SystemExit(f"Missing {CSV}")

data = pd.read_csv(CSV)
if 'timestamp' not in data.columns:
    raise SystemExit('timestamp column missing')

ts = pd.to_numeric(data['timestamp'], errors='coerce')
if ts.isna().any():
    raise SystemExit('timestamp has non-numeric')

elapsed = (ts - ts.iloc[0]).astype(float)

LEG_PREFIXES = ['FR','FL','RR','RL']
FOOT_COLUMNS = {'FL':'foot_FL','FR':'foot_FR','RL':'foot_RL','RR':'foot_RR'}

# compute per-leg metrics
results = {}
for leg in LEG_PREFIXES:
    # torque columns
    tau_cols = [f"{leg}_hip_tau", f"{leg}_thigh_tau", f"{leg}_calf_tau"]
    dq_cols = [f"{leg}_hip_dq", f"{leg}_thigh_dq", f"{leg}_calf_dq"]
    # safety: drop columns not present
    tau_cols = [c for c in tau_cols if c in data.columns]
    dq_cols = [c for c in dq_cols if c in data.columns]
    if not tau_cols or not dq_cols:
        continue
    torque_abs_sum = data[tau_cols].abs().sum(axis=1)
    vel_abs_mean = data[dq_cols].abs().mean(axis=1)
    foot_force = data[FOOT_COLUMNS[leg]] if FOOT_COLUMNS[leg] in data.columns else pd.Series(0, index=data.index)

    # normalize 0-1
    def norm(s):
        if s.max() == s.min():
            return (s - s.min())*0.0
        return (s - s.min()) / (s.max() - s.min())

    t_n = norm(torque_abs_sum)
    v_n = norm(vel_abs_mean)
    f_n = norm(foot_force)

    # entanglement score heuristic: high torque, low velocity, and foot in contact -> product
    ent = t_n * (1 - v_n) * (0.5 + 0.5 * f_n)  # weight foot factor to [0.5,1]

    # smooth
    window = max(3, int(len(ent) * 0.01))
    ent_smooth = ent.rolling(window=window, center=True, min_periods=1).mean()

    results[leg] = {
        'torque_abs_sum': torque_abs_sum,
        'vel_abs_mean': vel_abs_mean,
        'foot_force': foot_force,
        'ent_score': ent_smooth,
    }

# find intervals where ent_score exceeds threshold
TH = 0.6
intervals = {}
for leg, v in results.items():
    s = v['ent_score']
    above = s > TH
    intervals_list = []
    in_interval = False
    start_idx = None
    for i, val in enumerate(above):
        if val and not in_interval:
            in_interval = True
            start_idx = i
        if not val and in_interval:
            end_idx = i - 1
            intervals_list.append((start_idx, end_idx))
            in_interval = False
    if in_interval:
        intervals_list.append((start_idx, len(s)-1))
    # convert to time
    intervals_time = [(float(elapsed.iloc[a]), float(elapsed.iloc[b])) for a,b in intervals_list]
    intervals[leg] = intervals_time

# summary
print('Entanglement analysis for', CSV)
for leg in LEG_PREFIXES:
    if leg not in results:
        print(leg, ': missing data columns, skipped')
        continue
    score = results[leg]['ent_score']
    max_idx = int(score.idxmax())
    max_val = float(score.max())
    print(f"Leg {leg}: max ent_score={max_val:.3f} at t={float(elapsed.iloc[max_idx]):.3f}s")
    iv = intervals.get(leg, [])
    if iv:
        print('  Detected high-score intervals:')
        for a,b in iv:
            print(f"    {a:.3f}s -> {b:.3f}s")
    else:
        print('  No intervals exceed threshold')

# find most suspicious leg
max_leg = None
max_val = 0.0
for leg, v in results.items():
    mv = v['ent_score'].max()
    if mv > max_val:
        max_val = mv
        max_leg = leg

print('\nMost suspicious leg:', max_leg, 'score', f"{max_val:.3f}")

# save ent_score to CSV for inspection
out = Path('plots/go2_lowstate_1781770500/analysis')
out.mkdir(parents=True, exist_ok=True)
ent_df = pd.DataFrame({ 'elapsed': elapsed })
for leg, v in results.items():
    ent_df[f'ent_{leg}'] = v['ent_score'].values
ent_df.to_csv(out / 'ent_scores.csv', index=False)
print('Wrote', out / 'ent_scores.csv')
