#!/usr/bin/env python3
import argparse, pathlib, sys
import pandas as pd
import matplotlib.pyplot as plt

def human_usd(x): return f"{x:,.0f}"

def load_table(path):
    p = pathlib.Path(path)
    suffix = p.suffix.lower()
    # Excel
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p)
        except Exception as e:
            print("Need Excel support. Installing:", e, file=sys.stderr)
            raise
    # CSV with robust encodings
    for enc in ("utf-8", "cp1252", "latin-1", "utf-16", "utf-16le", "utf-16be"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort
    return pd.read_csv(p, engine="python", encoding_errors="ignore")

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--outdir", default="outputs/figures")
p.add_argument("--niat_col", default="NIAT")
p.add_argument("--year_col", default="year")
p.add_argument("--corr_vars", nargs="*", default=[
    "NIAT","ROA","ESG","E","S","G","SIZE","LEVERAGE","AGE","TANGIBILITY"
])
a = p.parse_args()

outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
df = load_table(a.data)

if a.niat_col not in df.columns:
    raise SystemExit(f"Column not found: {a.niat_col}. Available: {list(df.columns)[:30]} ...")

s = pd.to_numeric(df[a.niat_col], errors="coerce")
n_all = len(df); n_niat = s.notna().sum()
mean_ = s.mean(skipna=True); median_ = s.median(skipna=True)

print("Panel firm-years (all rows):", n_all)
print("NIAT non-missing N:", n_niat)
print("NIAT mean USD:", human_usd(mean_))
print("NIAT median USD:", human_usd(median_))

# Histograms (linear, symlog, trimmed)
sv = s.dropna()

# 1) Linear (current Figure 2)
plt.figure()
plt.hist(sv, bins=100)
plt.axvline(mean_, linestyle="--", linewidth=2, label=f"Mean: ${human_usd(mean_)}")
plt.axvline(median_, linestyle="-.", linewidth=2, label=f"Median: ${human_usd(median_)}")
plt.title("Distribution of Net Income After Taxes (NIAT)")
plt.xlabel("NIAT (USD)"); plt.ylabel("Frequency"); plt.legend()
plt.tight_layout(); plt.savefig(outdir / "figure_2_niat_histogram.png", dpi=300); plt.close()

# 2) Symmetric log x-axis (recommended replacement for Figure 2)
plt.figure()
plt.hist(sv, bins=150)
plt.axvline(mean_, linestyle="--", linewidth=2, label=f"Mean: ${human_usd(mean_)}")
plt.axvline(median_, linestyle="-.", linewidth=2, label=f"Median: ${human_usd(median_)}")
plt.xscale("symlog", linthresh=1e6)  # linear within ±$1m
plt.title("Distribution of NIAT (symlog x-axis)")
plt.xlabel("NIAT (USD, symmetric log scale)"); plt.ylabel("Frequency"); plt.legend()
plt.tight_layout(); plt.savefig(outdir / "figure_2_niat_histogram_symlog.png", dpi=300); plt.close()

# 3) Percentile-trimmed (1st–99th) for readability (appendix)
lo, hi = sv.quantile([0.01, 0.99])
sv_trim = sv[(sv >= lo) & (sv <= hi)]
plt.figure()
plt.hist(sv_trim, bins=120)
plt.axvline(sv_trim.mean(), linestyle="--", linewidth=2, label=f"Trimmed mean: ${human_usd(sv_trim.mean())}")
plt.axvline(sv_trim.median(), linestyle="-.", linewidth=2, label=f"Trimmed median: ${human_usd(sv_trim.median())}")
plt.title("Distribution of NIAT (1st–99th percentiles)")
plt.xlabel("NIAT (USD)"); plt.ylabel("Frequency"); plt.legend()
plt.tight_layout(); plt.savefig(outdir / "figure_2_niat_histogram_p01_p99.png", dpi=300); plt.close()


# Box-plots by year
if a.year_col in df.columns:
    g = df[[a.year_col, a.niat_col]].copy()
    g[a.niat_col] = pd.to_numeric(g[a.niat_col], errors="coerce")
    g = g.dropna(subset=[a.niat_col])
    years = sorted(pd.unique(g[a.year_col]))
    data = [g.loc[g[a.year_col]==y, a.niat_col].values for y in years]
    if len(data) > 0:
        plt.figure()
        plt.boxplot(data, labels=years, showfliers=True)
        plt.title("Box Plot of NIAT by Year")
        plt.xlabel("Year"); plt.ylabel("NIAT (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout(); plt.savefig(outdir / "figure_3_niat_boxplot_by_year.png", dpi=300); plt.close()

# Descriptives CSV
desc = pd.DataFrame({"N":[n_niat],"Mean_USD":[mean_],"Median_USD":[median_],
                     "StdDev_USD":[sv.std(ddof=1)],"Min_USD":[sv.min()],"Max_USD":[sv.max()]})
desc.to_csv(outdir / "table_niat_descriptives.csv", index=False)

# Correlations
corr_vars = [c for c in a.corr_vars if c in df.columns]
if corr_vars:
    df[corr_vars].apply(pd.to_numeric, errors="coerce").corr().to_csv(outdir / "table_correlation_matrix.csv")

print("Saved figures and tables to:", outdir)
