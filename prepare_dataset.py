
import os
import csv
import numpy as np
import pandas as pd

DATA_DIR = "sign_data"
OUT_CSV = os.path.join(DATA_DIR, "signs_63.csv")

def normalize_landmarks(flat):
    """flat = [x0,y0,z0,x1,y1,z1,...]. Normalize to wrist origin & scale."""
    arr = np.array(flat, dtype=float).reshape(-1, 3) 
    wrist = arr[0].copy()
    arr -= wrist  
    dists = np.linalg.norm(arr[:, :2], axis=1)
    scale = dists.max()
    if scale > 0:
        arr /= scale

    return arr.flatten().tolist()  

rows = [] 

for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for fname in os.listdir(label_path):
        if not fname.lower().endswith(".csv"):
            continue
        fpath = os.path.join(label_path, fname)
        try:
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                data_rows = list(reader)
            if not data_rows:
                continue
            raw = [float(v) for v in data_rows[0] if v.strip() != ""]
        except Exception as e:
            print(f"[SKIP] {fpath}: {e}")
            continue

        if len(raw) < 63:
            print(f"[SKIP] {fpath}: only {len(raw)} values (<63)")
            continue

        flat63 = raw[:63]

        norm63 = normalize_landmarks(flat63)
        rows.append(norm63 + [label.upper()])

print(f"[INFO] Loaded {len(rows)} usable samples.")

if not rows:
    raise SystemExit("No valid samples. Collect more data.")


num_feats = 63
cols = [f"x{i}" for i in range(num_feats)] + ["label"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv(OUT_CSV, index=False)
print(f"[DONE] Saved cleaned dataset to {OUT_CSV} with {len(df)} samples.")



