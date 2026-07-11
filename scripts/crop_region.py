import csv

import numpy as np

base = "/storage/HE_GPT/HEIR/artifacts/B1/"
rows = list(csv.DictReader(open(base + "B1_4_nuclei.csv")))
x = np.array([float(r["x"]) for r in rows])
y = np.array([float(r["y"]) for r in rows])
ids = [r["nucleus_id"] for r in rows]
cx, cy = np.median(x), np.median(y)
lo, hi = 0.0, float(max(x.max() - x.min(), y.max() - y.min()))
target = 15000
for _ in range(50):
    h = (lo + hi) / 2
    if ((np.abs(x - cx) <= h) & (np.abs(y - cy) <= h)).sum() > target:
        hi = h
    else:
        lo = h
m = (np.abs(x - cx) <= hi) & (np.abs(y - cy) <= hi)
sel = np.flatnonzero(m)[:16000]
print("region nuclei:", len(sel), "half-width px:", round(hi, 1), flush=True)
hdr = list(rows[0].keys())
with open(base + "B1_4_nuclei_region.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr)
    w.writeheader()
    for i in sel:
        w.writerow(rows[i])
d = np.load(base + "B1_4_features.npz")
fid = d["nucleus_ids"].astype(str)
pos = {v: i for i, v in enumerate(fid)}
order = [pos[ids[i]] for i in sel]
np.savez(
    base + "B1_4_features_region.npz",
    nucleus_ids=fid[order],
    features=d["features"][order],
    feature_names=d["feature_names"],
)
print("wrote region files:", len(order), flush=True)
