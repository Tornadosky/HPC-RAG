#!/usr/bin/env python3
"""
augment_hpc_framework_dataset.py
--------------------------------
Create synthetic user-profile rows for the HPC-framework recommender demo.

• Reads the 80-row CSV produced earlier.
• For every row it fabricates K variants (default 4).
• Slider columns get small Gaussian noise.
• Discrete columns (gpu_skill_level, project type, domain, prefs) may flip
  with controlled probabilities.
• Vendor-specific hardware flags are *never* invalidated for that framework.
• `need_cross_vendor` is recomputed from the new state.

Author: <you>
"""

import argparse
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Configuration knobs (can also be overriden from the CLI if you like)
# -----------------------------------------------------------------------------#
SLIDER_SIGMA          = 0.06   # std-dev for perf/port/eco/lockin sliders
SKILL_PROB_SHIFT      = 0.30   # 30 % chance to ±1 gpu_skill_level
PREF_FLIP_PROB        = 0.15   # chance to swap directive <-> kernel comfort
PROJECT_FLIP_PROB     = 0.12   # chance to switch greenfield|gpu_extend|cpu_port
DOMAIN_FLIP_PROB      = 0.20   # chance to replace the active domain
HW_TOGGLE_PROB        = 0.10   # chance to toggle an *extra* hw flag (if allowed)

# column groups
SLIDER_COLS  = ["perf_weight", "port_weight", "eco_weight", "lockin_tolerance"]
PROJECT_COLS = ["greenfield", "gpu_extend", "cpu_port"]
DOMAIN_COLS  = [
    "domain_ai_ml", "domain_hpc", "domain_climate", "domain_embedded",
    "domain_graphics", "domain_data_analytics", "domain_other",
]
HW_COLS      = ["hw_cpu", "hw_nvidia", "hw_amd", "hw_other"]

# -----------------------------------------------------------------------------#


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="infile",  default="training_80.csv",
                    help="input CSV (original 80 rows)")
    ap.add_argument("--out", dest="outfile", default="training_80_aug.csv",
                    help="output CSV for augmented data")
    ap.add_argument("-k", "--variants", type=int, default=4,
                    help="how many synthetic rows per original row")
    return ap.parse_args()


def vendor_required_flags(framework: str) -> dict:
    """
    Return the *minimum* hardware flags that must remain 1 for this framework.
    """
    if framework == "CUDA":
        return {"hw_nvidia": 1}
    if framework == "HIP":
        return {"hw_amd": 1}
    # others: at least CPU stays 1 for realism
    return {"hw_cpu": 1}


def pick_other_onehot(cols, rng):
    """Return a new one-hot vector with exactly one 1 among the given cols."""
    chosen = rng.choice(cols)
    return {c: int(c == chosen) for c in cols}


def recompute_cross_vendor(row: pd.Series) -> int:
    """Set need_cross_vendor according to lock-in & multiple hardware targets."""
    multi_hw = row[HW_COLS].sum() > 1
    low_lock = row["lockin_tolerance"] < 0.35
    return int(multi_hw and low_lock)


def augment_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Return *one* synthetic variant of the input row."""
    new = row.copy()

    # 1. jitter sliders --------------------------------------------------------
    noise = rng.normal(0, SLIDER_SIGMA, size=len(SLIDER_COLS))
    new[SLIDER_COLS] = np.clip(new[SLIDER_COLS] + noise, 0, 1)

    # 2. gpu_skill_level shift -------------------------------------------------
    if rng.random() < SKILL_PROB_SHIFT:
        delta = rng.choice([-1, +1])
        new["gpu_skill_level"] = int(np.clip(new["gpu_skill_level"] + delta, 0, 3))

    # 3. maybe flip directive/kernel comfort ----------------------------------
    if rng.random() < PREF_FLIP_PROB:
        new["pref_directives"], new["pref_kernels"] = \
            new["pref_kernels"], new["pref_directives"]

    # 4. maybe switch project type --------------------------------------------
    if rng.random() < PROJECT_FLIP_PROB:
        new.update(pick_other_onehot(PROJECT_COLS, rng))

    # 5. maybe switch domain ---------------------------------------------------
    if rng.random() < DOMAIN_FLIP_PROB:
        new.update(pick_other_onehot(DOMAIN_COLS, rng))

    # 6. maybe toggle an extra hardware flag (but honour vendor constraints) ---
    required = vendor_required_flags(new["framework"])
    allowed_cols = [c for c in HW_COLS if c not in required]
    if allowed_cols and rng.random() < HW_TOGGLE_PROB:
        col = rng.choice(allowed_cols)
        new[col] = 1 - new[col]

    # 7. ensure required hardware flags are still 1 ----------------------------
    for col, val in required.items():
        new[col] = val

    # 8. recompute need_cross_vendor ------------------------------------------
    new["need_cross_vendor"] = recompute_cross_vendor(new)

    return new


def main():
    args = parse_args()
    rng = np.random.default_rng(seed=2025)

    base = pd.read_csv(args.infile)
    augmented = [base]

    synthetic_rows = []
    for idx, row in base.iterrows():
        for k in range(args.variants):
            synth = augment_row(row, rng)
            synth["user_id"] = f"{row['user_id']}_S{k+1}"
            synthetic_rows.append(synth)

    # build a DataFrame from all of the synthetic Series at once
    synth_df = pd.DataFrame(synthetic_rows)

    # now concat two DataFrames (base + synth_df), not Series
    out_df = pd.concat([base, synth_df], ignore_index=True)
    out_df.to_csv(args.outfile, index=False, float_format="%.2f")
    print(f"Augmented data saved → {args.outfile}  ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
