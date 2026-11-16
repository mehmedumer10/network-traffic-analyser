# -*- coding: utf-8 -*-
"""
generate_sample.py
------------------
Generates realistic network-traffic CSV files that match the
8 features your LightGBM model was trained on:

    1. Avg Packet Size
    2. Packet Length Mean
    3. Avg Fwd Segment Size
    4. Fwd Packet Length Mean
    5. Fwd Packets Length Total
    6. Packet Length Min
    7. Fwd Packet Length Min
    8. Packet Length Max

Label encoding:  0 = Attack (Malicious)   1 = Benign (Normal)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. SELECTED FEATURES (must be EXACTLY the same as training)
# ------------------------------------------------------------------
SELECTED_FEATURES = [
    'Avg Packet Size',
    'Packet Length Mean',
    'Avg Fwd Segment Size',
    'Fwd Packet Length Mean',
    'Fwd Packets Length Total',
    'Packet Length Min',
    'Fwd Packet Length Min',
    'Packet Length Max'          # <-- replaces the old "Subflow Fwd Bytes"
]

# ------------------------------------------------------------------
# 2. INTERACTIVE MODE SELECTION
# ------------------------------------------------------------------
def get_user_mode():
    print("\n" + "="*60)
    print("   NETWORK TRAFFIC DATASET GENERATOR")
    print("="*60)
    print("Label Encoding: 0 = Attack (Malicious), 1 = Benign (Normal)")
    print("\nChoose dataset mode:")
    print("   1) Both (Benign + Attack)")
    print("   2) Benign only")
    print("   3) Attack only")
    print("   4) All three (recommended)")
    print("-"*60)

    while True:
        choice = input("Enter choice (1/2/3/4): ").strip()
        if choice == '1':  return ["both"]
        if choice == '2':  return ["benign_only"]
        if choice == '3':  return ["attack_only"]
        if choice == '4':  return ["both", "benign_only", "attack_only"]
        print("Invalid choice – please enter 1, 2, 3 or 4.")

MODES_TO_GENERATE = get_user_mode()
print(f"\nYou selected: {', '.join(m.title().replace('_', ' ') for m in MODES_TO_GENERATE)}\n")

# ------------------------------------------------------------------
# 3. SYNTHETIC DATA GENERATOR (realistic distributions)
# ------------------------------------------------------------------
def generate_synthetic_dataset(n_rows: int = 1_000_000) -> pd.DataFrame:
    """
    Creates a balanced synthetic dataset (60 % benign, 40 % attack)
    with realistic means / std-devs for the 8 selected features.
    """
    n_benign = int(n_rows * 0.60)
    n_attack = n_rows - n_benign

    # ---- statistical profiles (mean, std) ----
    benign_profile = {
        'Avg Packet Size':          (400, 100),
        'Packet Length Mean':       (420, 110),
        'Avg Fwd Segment Size':     (400, 100),
        'Fwd Packet Length Mean':   (420, 110),
        'Fwd Packets Length Total': (2000, 500),
        'Packet Length Min':        (60,  20),
        'Fwd Packet Length Min':    (60,  20),
        'Packet Length Max':        (1500, 300)   # realistic Ethernet MTU
    }

    attack_profile = {
        'Avg Packet Size':          (800, 200),
        'Packet Length Mean':       (850, 220),
        'Avg Fwd Segment Size':     (800, 200),
        'Fwd Packet Length Mean':   (850, 220),
        'Fwd Packets Length Total': (5000, 1000),
        'Packet Length Min':        (40,  15),
        'Fwd Packet Length Min':    (40,  15),
        'Packet Length Max':        (1500, 200)   # attacks often hit MTU
    }

    data = {}
    for feat in SELECTED_FEATURES:
        b = np.random.normal(*benign_profile[feat], n_benign)
        a = np.random.normal(*attack_profile[feat], n_attack)
        data[feat] = np.concatenate([b, a])

    df = pd.DataFrame(data)

    # ---- realistic inter-feature relationships ----
    n = len(df)
    df['Packet Length Mean']   = df['Avg Packet Size'] * (1 + np.random.normal(0.05, 0.10, n))
    df['Fwd Packet Length Mean'] = df['Packet Length Mean'] * (1 + np.random.normal(0.00, 0.05, n))
    df['Avg Fwd Segment Size'] = df['Avg Packet Size'] * (1 + np.random.normal(0.02, 0.08, n))

    # Total forward bytes = avg packet size × #packets (3-15 packets per flow)
    df['Fwd Packets Length Total'] = df['Avg Packet Size'] * np.random.randint(3, 16, n)

    # Packet Length Max = max of (min + random jitter, MTU)
    jitter = np.random.uniform(0, 300, n)
    df['Packet Length Max'] = np.maximum(df['Packet Length Min'] + jitter, 1500)

    # ---- enforce non-negative values ----
    df[SELECTED_FEATURES] = df[SELECTED_FEATURES].abs()

    # ---- add some extreme outliers to attack traffic (20 % of attacks) ----
    attack_idx = df.index[-n_attack:]
    outlier_idx = np.random.choice(attack_idx, size=int(n_attack * 0.20), replace=False)
    for col in ['Fwd Packets Length Total', 'Packet Length Max']:
        df.loc[outlier_idx, col] *= np.random.uniform(3, 10, len(outlier_idx))

    # ---- labels ----
    df['label'] = np.concatenate([np.ones(n_benign), np.zeros(n_attack)]).astype(int)

    # ---- round to 2 decimals (matches your model input) ----
    df[SELECTED_FEATURES] = df[SELECTED_FEATURES].round(2)

    return df[SELECTED_FEATURES + ['label']]

# ------------------------------------------------------------------
# 4. FILTER BY MODE
# ------------------------------------------------------------------
def filter_dataset(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    feats = SELECTED_FEATURES
    if mode == "benign_only":
        out = df[df['label'] == 1].copy()
        print(f"   → Benign only : {len(out):,} rows")
    elif mode == "attack_only":
        out = df[df['label'] == 0].copy()
        print(f"   → Attack only : {len(out):,} rows")
    else:  # both
        out = df.copy()
        b = out['label'].sum()
        a = len(out) - b
        print(f"   → Both        : {len(out):,} rows (Benign {b:,}, Attack {a:,})")
    return out[feats + ['label']]

# ------------------------------------------------------------------
# 5. MAIN EXECUTION
# ------------------------------------------------------------------
print("\nGenerating base dataset (1 000 000 rows)…")
base_df = generate_synthetic_dataset(n_rows=1_000_000)

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(output_dir, exist_ok=True)

suffix = {
    "both":         "with_labels",
    "benign_only":  "benign_only",
    "attack_only":  "attack_only"
}

generated = []   # (mode, full_path, small_path, rows)

for mode in MODES_TO_GENERATE:
    print(f"\nGenerating → {mode.upper().replace('_', ' ')}")
    df_f = filter_dataset(base_df, mode)

    # full file
    full_name = f"network_traffic_{suffix[mode]}.csv"
    full_path = os.path.join(output_dir, full_name)
    df_f.to_csv(full_path, index=False)

    # small demo file (first 200 rows)
    small_name = f"network_traffic_small_{suffix[mode]}.csv"
    small_path = os.path.join(output_dir, small_name)
    df_f.head(200).to_csv(small_path, index=False)

    generated.append((mode, full_name, small_name, len(df_f)))
    print(f"   Saved full : {full_path}")
    print(f"   Saved small: {small_path}")

# ------------------------------------------------------------------
# 6. FINAL SUMMARY
# ------------------------------------------------------------------
print("\n" + "="*80)
print("ALL DATASETS GENERATED SUCCESSFULLY!")
print("="*80)
for _, full, small, rows in generated:
    print(f"{full:<38} → {rows:,} rows")
print(f"\nOutput folder: {output_dir}")
print("\nLabel meaning:")
print("   0 → Attack (Malicious)")
print("   1 → Benign (Normal)")

print("\nFiles created:")
for _, full, small, _ in generated:
    print(f"   • {full}")
    print(f"   • {small}")

print("\n" + "READY FOR MODEL TESTING!")
print("="*80)