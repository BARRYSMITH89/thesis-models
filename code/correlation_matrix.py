# code/correlation_matrix.py
# Build a correlation matrix and heatmap from the current dataset.
# Default CSV: data/Final Data Including Controls.csv
# Usage:
#   python3 code/correlation_matrix.py
#   python3 code/correlation_matrix.py "/path/to/Final Data Including Controls.csv"

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_DEFAULT = "Final Data Including Controls.csv"
VARS = [
    "ROA", "NIAT",
    "Environmental_Score", "Social_Score", "Governance_Score", "ESG_Score",
    "Employees", "Assets", "Debt"
]

def read_any(path: Path) -> pd.DataFrame:
    for enc in ("cp1252", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read CSV at: {path}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / CSV_DEFAULT
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        print(f"[HINT] Put '{CSV_DEFAULT}' in {default_csv.parent} or pass a full path.")
        sys.exit(1)

    df = read_any(csv_path)

    # Pairwise-complete correlations (pandas default)
    corr = df[VARS].corr().round(2)

    # Write table
    out_tables = project_root / "outputs" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_tables / "correlation_matrix.csv", index=True)

    # Heatmap with numeric annotations (matplotlib only)
    out_figs = project_root / "outputs" / "figures"
    out_figs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(VARS))); ax.set_yticks(range(len(VARS)))
    ax.set_xticklabels(VARS, rotation=90)
    ax.set_yticklabels(VARS)

    # annotate
    for i in range(len(VARS)):
        for j in range(len(VARS)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im)
    cbar.set_label("Pearson r", rotation=90)
    ax.set_title("Correlation Matrix of Key Variables (pairwise deletion)")
    fig.tight_layout()
    fig.savefig(out_figs / "correlation_matrix.png", dpi=300)
    plt.close(fig)

    print("[OK] Wrote:")
    print(f"  - {out_tables/'correlation_matrix.csv'}")
    print(f"  - {out_figs/'correlation_matrix.png'}")
    print(f"[CSV used] {csv_path.resolve()}")

if __name__ == "__main__":
    main()
