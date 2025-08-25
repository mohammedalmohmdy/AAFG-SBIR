import glob, json, numpy as np, pandas as pd
import argparse

def bootstrap_ci(x, iters=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = []
    for _ in range(iters):
        sample = rng.choice(x, size=n, replace=True)
        bs.append(np.mean(sample))
    lo = np.percentile(bs, 100*alpha/2)
    hi = np.percentile(bs, 100*(1-alpha/2))
    return float(lo), float(hi)

def summarize(files):
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    rows = []
    group_cols = [c for c in ['dataset','method'] if c in df.columns]
    for key, g in df.groupby(group_cols):
        if isinstance(key, tuple):
            dataset, method = key
        else:
            dataset, method = (g['dataset'].iloc[0], key)
        row = dict(dataset=dataset, method=method)
        for metric in [c for c in ['mAP','P@100','P@200'] if c in g.columns]:
            vals = g[metric].to_numpy(dtype=float)
            mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1))
            lo, hi = bootstrap_ci(vals, seed=0)
            row[metric] = {
              "mean": round(mean, 3),
              "std": round(std, 3),
              "ci95": [round(lo,3), round(hi,3)],
              "n": len(vals)
            }
        rows.append(row)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for per-seed CSVs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = glob.glob(args.glob)
    if not files:
        raise SystemExit("No CSVs matched!")
    report = summarize(files)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out} with {len(report)} rows.")
