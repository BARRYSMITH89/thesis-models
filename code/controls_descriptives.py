# code/controls_summary.py
# Summarises control variables for the 20,995-row panel:
# Employees, Debt, Assets, Year, and Leverage = Debt/Assets.
# Outputs CSV tables and basic figures.

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DEFAULT = "Final Data Including Controls.csv"
CONTROLS = ["Employees", "Debt", "Assets", "Year"]

def read_any(path: Path) -> pd.DataFrame:
    for enc in ("cp1252", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read CSV at: {path}")

def describe(s: pd.Series, total_n: int) -> dict:
    x = s.dropna()
    return {
        "n": int(x.size),
        "coverage_pct": x.size / total_n * 100.0,
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "q1": float(x.quantile(0.25)),
        "median": float(x.median()),
        "q3": float(x.quantile(0.75)),
        "max": float(x.max()),
    }

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / CSV_DEFAULT
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Put '{CSV_DEFAULT}' in {default_csv.parent} or pass a full path.")
        sys.exit(1)

    df = read_any(csv_path)
    total = len(df)

    out_tables = ensure_dir(project_root / "outputs" / "tables")
    out_figs   = ensure_dir(project_root / "outputs" / "figures")

    # 1) Availability
    avail = df[CONTROLS].notna().sum()
    availability = pd.DataFrame({
        "variable": avail.index,
        "non_null": avail.values,
        "missing": total - avail.values,
        "pct_available": (avail.values / total * 100.0),
    }).round(2)
    availability.to_csv(out_tables / "controls_availability.csv", index=False)

    # 2) Descriptives for Employees, Debt, Assets, Year
    rows = []
    for col in CONTROLS:
        rows.append({"variable": col, **describe(df[col], total)})
    desc = pd.DataFrame(rows).round(2)
    desc.to_csv(out_tables / "controls_descriptives.csv", index=False)

    # 3) Leverage = Debt/Assets when both present and Assets != 0
    mask = df["Debt"].notna() & df["Assets"].notna() & (df["Assets"] != 0)
    leverage = (df.loc[mask, "Debt"] / df.loc[mask, "Assets"]).astype(float)
    lev_desc = pd.DataFrame([describe(leverage, total)]).round(4)
    lev_desc.insert(0, "variable", "Leverage(Debt/Assets)")
    lev_desc.to_csv(out_tables / "leverage_descriptives.csv", index=False)

    # 4) Basic figures (hist + box) for Employees, Debt, Assets, Leverage
    for name, s in {
        "Employees": df["Employees"].dropna(),
        "Debt": df["Debt"].dropna(),
        "Assets": df["Assets"].dropna(),
        "Leverage": leverage.dropna(),
    }.items():
        # Histogram
        plt.figure()
        plt.hist(s, bins=30)
        plt.title(f"{name} distribution")
        plt.xlabel(name); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(out_figs / f"{name}_hist.png", dpi=300); plt.close()

        # Boxplot
        plt.figure()
        plt.boxplot(s, vert=True, showfliers=True)
        plt.title(f"{name} boxplot"); plt.ylabel(name)
        plt.tight_layout(); plt.savefig(out_figs / f"{name}_box.png", dpi=300); plt.close()

    # Console check
    print("[OK] Tables written to:", out_tables)
    print("[OK] Figures written to:", out_figs)
    print(f"[CSV used] {csv_path.resolve()}")

if __name__ == "__main__":
    main()
