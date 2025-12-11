# code/describe_pillars.py
# Outputs descriptives for ESG and the E/S/G pillars, availability, correlations,
# sector and size splits, top firms list (names only), yearly trends, and figures.

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DEFAULT_NAME = "Final Data Including Controls.csv"
PILLARS = ["Environmental_Score", "Social_Score", "Governance_Score"]
ESG = "ESG_Score"

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("cp1252", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV at: {path}")

def describe(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "n": int(s.size),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    # Resolve paths
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / CSV_DEFAULT_NAME
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Put '{CSV_DEFAULT_NAME}' in {default_csv.parent} or pass an explicit path.")
        sys.exit(1)

    df = read_csv_any(csv_path)
    out_tables = ensure_dir(project_root / "outputs" / "tables")
    out_figs = ensure_dir(project_root / "outputs" / "figures")

    # 1) Availability
    cols = PILLARS + [ESG]
    total = len(df)
    avail = df[cols].notna().sum()
    availability = pd.DataFrame({
        "variable": avail.index,
        "non_null": avail.values,
        "missing": total - avail.values,
        "pct_available": (avail.values / total * 100),
    }).round(2)
    availability.to_csv(out_tables / "variable_availability.csv", index=False)

    # 2) Descriptives: ESG + pillars
    rows = []
    for c in [ESG] + PILLARS:
        stats = describe(df[c])
        stats["variable"] = c
        rows.append(stats)
    desc = pd.DataFrame(rows)[["variable","n","mean","std","min","q1","median","q3","max"]].round(2)
    desc.to_csv(out_tables / "esg_and_pillars_descriptives.csv", index=False)

    # 3) Correlations (pairwise complete)
    corr = df[[ESG] + PILLARS].dropna().corr().round(2)
    corr.to_csv(out_tables / "pillar_correlations.csv")

    # 4) Sector means (if column exists)
    if "Sector" in df.columns:
        sector_means = (
            df[[ "Sector", ESG] + PILLARS]
            .dropna(subset=[ESG])
            .groupby("Sector", as_index=False)
            .mean()
            .round(2)
            .sort_values(ESG, ascending=False)
        )
        sector_means.to_csv(out_tables / "sector_means_esg_pillars.csv", index=False)

    # 5) Size split by median Employees (if column exists)
    if "Employees" in df.columns:
        sized = df[[ESG] + PILLARS + ["Employees"]].dropna(subset=[ESG, "Employees"]).copy()
        med_emp = sized["Employees"].median()
        sized["size_group"] = np.where(sized["Employees"] >= med_emp, "Large (>= median)", "Small (< median)")
        size_means = sized.groupby("size_group")[[ESG] + PILLARS].mean().round(2).reset_index()
        size_means.to_csv(out_tables / "size_split_means_esg_pillars.csv", index=False)
        Path(out_tables / "size_split_median_employees.txt").write_text(f"median_employees={med_emp}\n")

    # 6) Top and bottom firms by average ESG (names only)
    if "Company" in df.columns:
        by_firm = (
            df[["Company", ESG]]
            .dropna(subset=[ESG])
            .groupby("Company", as_index=False)[ESG]
            .mean()
            .sort_values(ESG, ascending=False)
        )
        top_names = by_firm.head(5)[["Company"]]
        bot_names = by_firm.tail(5)[["Company"]]
        top_names.to_csv(out_tables / "top5_companies_by_avg_esg_names_only.csv", index=False)
        bot_names.to_csv(out_tables / "bottom5_companies_by_avg_esg_names_only.csv", index=False)
        # Detailed (optional, for internal checks)
        by_firm.round(2).to_csv(out_tables / "companies_by_avg_esg_with_scores.csv", index=False)

    # 7) Yearly trend for ESG and pillars + figure
    if "Year" in df.columns:
        trend = (
            df[["Year", ESG] + PILLARS]
            .dropna(subset=[ESG])
            .groupby("Year", as_index=False)
            .mean()
            .round(2)
            .sort_values("Year")
        )
        trend.to_csv(out_tables / "trend_by_year_esg_pillars.csv", index=False)

        # Single plot with four lines, no style/colour settings
        plt.figure()
        for col in [ESG] + PILLARS:
            plt.plot(trend["Year"], trend[col], label=col)
        plt.title("Average ESG and pillar scores by year")
        plt.xlabel("Year")
        plt.ylabel("Average score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "trend_by_year_esg_pillars.png", dpi=300)
        plt.close()

    # 8) Simple histograms and boxplots for each metric (one plot per image)
    for col in [ESG] + PILLARS:
        s = df[col].dropna()
        # Histogram
        plt.figure()
        plt.hist(s, bins=20)
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_figs / f"{col}_hist.png", dpi=300)
        plt.close()

        # Boxplot
        plt.figure()
        plt.boxplot(s, vert=True, showfliers=True)
        plt.title(f"{col} boxplot")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out_figs / f"{col}_box.png", dpi=300)
        plt.close()

    print("[OK] Tables written to:", out_tables)
    print("[OK] Figures written to:", out_figs)
    print(f"[CSV used] {csv_path.resolve()}")

if __name__ == "__main__":
    main()
