# code/describe_esg.py
# Computes descriptives for ESG_Score and pillar scores.
# Default CSV location: ../data/Final Data Including Controls.csv
# Usage:
#   python3 code/describe_esg.py
#   python3 code/describe_esg.py "/custom/path/Final Data Including Controls.csv"

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

CSV_DEFAULT_NAME = "Final Data Including Controls.csv"
PILLARS = ["Environmental_Score", "Social_Score", "Governance_Score"]
ESG_COL = "ESG_Score"

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("cp1252", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV with common encodings at: {path}")

def describe_series(s: pd.Series) -> dict:
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

def main():
    # Resolve project root from this file: code/ -> project root
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / CSV_DEFAULT_NAME

    # Allow override via CLI
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Place '{CSV_DEFAULT_NAME}' in {default_csv.parent} "
              f"or run: python3 code/describe_esg.py \"/full/path/{CSV_DEFAULT_NAME}\"")
        sys.exit(1)

    df = read_csv_any(csv_path)

    # Outputs dir at project root
    outdir = project_root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # Availability table
    cols = PILLARS + [ESG_COL]
    avail = df[cols].notna().sum().rename("non_null")
    total = len(df)
    availability = pd.DataFrame({
        "variable": avail.index,
        "non_null": avail.values,
        "missing": total - avail.values,
        "pct_available": (avail.values / total * 100),
    }).round(2)
    availability.to_csv(outdir / "variable_availability.csv", index=False)

    # Combined ESG descriptives
    esg_desc = describe_series(df[ESG_COL])
    esg_summary = {
        "n_total_rows": int(total),
        "n_with_ESG": esg_desc["n"],
        "n_missing_ESG": int(total - esg_desc["n"]),
        **{k: v for k, v in esg_desc.items() if k != "n"},
    }
    pd.DataFrame([esg_summary]).round(2).to_csv(outdir / "esg_combined_summary.csv", index=False)

    # Pillar descriptives (useful for the next paragraphs)
    desc_rows = []
    for col in PILLARS:
        d = describe_series(df[col])
        d["variable"] = col
        desc_rows.append(d)
    pd.DataFrame(desc_rows).round(2).to_csv(outdir / "pillar_descriptives.csv", index=False)

    # Console output (concise)
    pretty = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in esg_summary.items()}
    print("[OK] Wrote:")
    print(f"  - {outdir/'esg_combined_summary.csv'}")
    print(f"  - {outdir/'variable_availability.csv'}")
    print(f"  - {outdir/'pillar_descriptives.csv'}")
    print("[ESG summary]", pretty)
    print(f"[CSV used] {csv_path.resolve()}")

if __name__ == "__main__":
    main()

