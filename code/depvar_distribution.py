# code/depvar_distribution.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarise(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n": int(s.size),
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(ddof=1),
        "min": s.min(),
        "max": s.max(),
        "p1": s.quantile(0.01),
        "p99": s.quantile(0.99),
    }

def read_csv_robust(path: str, encoding_hint: str | None):
    tried = [encoding_hint or "utf-8", "utf-8-sig", "latin1", "cp1252"]
    last_err = None
    for enc in tried:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--var", default="ROA")
    ap.add_argument("--outdir", default="outputs/descriptives")
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--encoding", default=None)  # e.g., latin1
    args = ap.parse_args()

    df = read_csv_robust(args.data, args.encoding)
    if args.var not in df.columns:
        raise SystemExit(f"Column '{args.var}' not found. Available: {list(df.columns)[:20]} ...")

    stats = summarise(df[args.var])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats], index=[args.var]).to_csv(outdir / f"{args.var.lower()}_summary.csv", float_format="%.6f")

    s = pd.to_numeric(df[args.var], errors="coerce").dropna()
    lo, hi = stats["p1"], stats["p99"]
    s_clip = s.clip(lower=lo, upper=hi)

    plt.figure()
    plt.hist(s_clip, bins=args.bins)
    plt.axvline(stats["mean"], linestyle="--", linewidth=1.5, label=f"Mean {stats['mean']:.2f}")
    plt.axvline(stats["median"], linestyle="-.", linewidth=1.5, label=f"Median {stats['median']:.2f}")
    plt.legend()
    plt.title(f"Distribution of {args.var}")
    plt.xlabel(args.var)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / f"{args.var.lower()}_hist.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

