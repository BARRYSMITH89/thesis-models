"""
VIF diagnostics for thesis panel models.
Reads:  data/Final Data Including Controls.csv
Writes: outputs/vif_summary.csv
"""

from pathlib import Path
import re
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Final Data Including Controls.csv"
OUT  = ROOT / "outputs" / "vif_summary.csv"

# ---------- helpers

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def find_col(df: pd.DataFrame, candidates):
    m = {_norm(c): c for c in df.columns}
    for c in candidates:
        k = _norm(c)
        if k in m:
            return m[k]
    raise KeyError(f"Missing any of {candidates}. Columns: {list(df.columns)}")

def hotel_mask(series: pd.Series):
    return series.astype(str).str.contains(r"\bhotel(s)?\b", case=False, na=False)

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    X = X.dropna().copy()
    arr = X.to_numpy()
    rows = []
    for i, name in enumerate(X.columns):
        rows.append((name, float(variance_inflation_factor(arr, i))))
    return pd.DataFrame(rows, columns=["variable", "vif"])

def compute_block(df, sample, lag, spec, cols):
    tab = vif_table(df[cols])
    tab.insert(0, "spec", spec)
    tab.insert(0, "lag", lag)
    tab.insert(0, "sample", sample)
    tab["mean_vif"] = tab["vif"].mean()
    tab["max_vif"]  = tab["vif"].max()
    return tab

def load_csv(path: Path) -> pd.DataFrame:
    # Try UTF-8, then Windows-1252 (cp1252), then Latin-1.
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except UnicodeDecodeError:
            continue
    # Last resort: replace bad bytes to avoid crash.
    return pd.read_csv(path, encoding="utf-8", engine="python", encoding_errors="replace")

# ---------- main

df = load_csv(DATA)

# column detection
firm   = find_col(df, ["Company", "Firm", "Organisation", "Org", "Ticker"])
year   = find_col(df, ["Year", "FiscalYear", "FY"])
sector = find_col(df, ["Sector", "Industry", "GICS_Sector"])
esg    = find_col(df, ["ESG_Score", "ESG", "ESG score"])
emp    = find_col(df, ["Employees", "EmployeeCount", "Headcount"])
debt   = find_col(df, ["Debt", "TotalDebt", "Leverage"])

# pillars if available
pillars = []
seen = set()
for cand in ["Environmental","E","Social","S","Governance","G","Env","Soc","Gov"]:
    try:
        col = find_col(df, [cand])
        initial = col.upper()[0]
        if initial not in seen:
            pillars.append(col)
            seen.add(initial)
    except KeyError:
        pass
has_pillars = seen >= {"E","S","G"}

df = df.sort_values([firm, year]).copy()

out = []
for sample_name, dsub in [("full", df), ("hotels", df[hotel_mask(df[sector])].copy())]:
    for lag in range(4):  # 0..3
        d = dsub.copy()
        d[f"{esg}_lag"] = d.groupby(firm)[esg].shift(lag)
        esg_cols = [f"{esg}_lag", emp, debt]
        tmp = d.dropna(subset=esg_cols)
        if len(tmp):
            out.append(compute_block(tmp, sample_name, lag, "ESG", esg_cols))
        if has_pillars:
            dd = dsub.copy()
            lag_cols = []
            for p in pillars:
                dd[f"{p}_lag"] = dd.groupby(firm)[p].shift(lag)
                lag_cols.append(f"{p}_lag")
            pill_cols = lag_cols + [emp, debt]
            tmp2 = dd.dropna(subset=pill_cols)
            if len(tmp2):
                out.append(compute_block(tmp2, sample_name, lag, "Pillars", pill_cols))

if not out:
    raise SystemExit("No rows available to compute VIFs. Check column names/mapping.")

res = pd.concat(out, ignore_index=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
