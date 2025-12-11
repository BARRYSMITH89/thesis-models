# hausman_appendix_A1.py
# Produce Appendix A1 Hausman FE vs RE for full hospitality and Hotels-only.

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from linearmodels.panel import PanelOLS, RandomEffects

CSV_FILE = Path((Path(__file__).resolve().parents[1] / 'data' / 'Final Data Including Controls.csv'))

# ---------- loaders ----------
def load_csv_any_encoding(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except UnicodeDecodeError:
            pass
    raise UnicodeError("Could not decode CSV.")

def resolve_columns(df: pd.DataFrame) -> dict:
    cols = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        norm = {re.sub(r"[^a-z]", "", k): v for k, v in cols.items()}
        for n in names:
            k2 = re.sub(r"[^a-z]", "", n.lower())
            if k2 in norm: return norm[k2]
        for k, v in cols.items():
            for n in names:
                if n.lower() in k: return v
        raise KeyError(f"Missing any of {names}")
    return {
        "firm":  pick(["Company","Firm","Organisation","Organization","Org","GVKEY","Ticker","ISIN","CompanyName"]),
        "year":  pick(["Year","FY","FiscalYear","Fiscal_Year","fyear"]),
        "sector":pick(["Sector","Industry","GICS","SIC_Desc","Subsector","Sub-industry"]),
        "roa":   pick(["ROA"]),
        "niat":  pick(["NIAT","Net_Income_After_Tax","NetIncome","NI"]),
        "esg":   pick(["ESG_Score","ESG","ESGTotal","ESGCombined","Refinitiv_ESG"]),
        "emp":   pick(["Employees","EmployeeCount","Headcount"]),
        "debt":  pick(["Debt","TotalDebt","Leverage","Total_Debt"]),
    }

# ---------- stats ----------
def hausman_stat(fe_res, re_res, keep_idx):
    b_fe, b_re = fe_res.params[keep_idx], re_res.params[keep_idx]
    V_fe, V_re = fe_res.cov.loc[keep_idx, keep_idx], re_res.cov.loc[keep_idx, keep_idx]
    d = (b_fe - b_re).values
    V = (V_fe - V_re).values
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(V)
    stat = float(d.T @ Vinv @ d)
    df = int(len(keep_idx))
    pval = float(1 - chi2.cdf(stat, df))
    return stat, df, pval

def build_design(df_idx: pd.DataFrame, y_col: str, esg_col: str, emp_col: str, debt_col: str, lag: int):
    df = df_idx.copy()
    df["ESG_lag"] = df[esg_col].groupby(level=0).shift(lag)
    X = df[["ESG_lag", emp_col, debt_col]].copy()
    y = df[y_col]
    keep = ~(y.isna() | X.isna().any(axis=1))
    y, X = y[keep], X[keep]
    # drop constant columns
    const_cols = [c for c in X.columns if np.isclose(X[c].std(ddof=0), 0.0)]
    if const_cols:
        X = X.drop(columns=const_cols)
    return y, X

def run_block(panel: pd.DataFrame, labels: dict, sample_label: str) -> pd.DataFrame:
    rows = []
    for out_name, ycol in [("ROA", labels["roa"]), ("NIAT", labels["niat"])]:
        for lag in range(0, 4):
            y, X = build_design(panel, ycol, labels["esg"], labels["emp"], labels["debt"], lag)
            if len(y) == 0 or X.shape[1] == 0:
                rows.append({"sample": sample_label, "outcome": out_name, "lag": lag,
                             "chi2": np.nan, "df": 0, "p": np.nan, "decision": "NA"})
                continue
            # FE: entity + time effects; drop absorbed cols automatically
            fe_res = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True)\
                        .fit(cov_type="unadjusted")
            # RE: same X (no explicit time dummies)
            re_res = RandomEffects(y, X).fit(cov_type="unadjusted")
            keep_idx = list(X.columns)  # compare only ESG_lag, Employees, Debt
            chi2v, df, p = hausman_stat(fe_res, re_res, keep_idx)
            rows.append({
                "sample": sample_label, "outcome": out_name, "lag": lag,
                "chi2": round(chi2v, 4), "df": df, "p": float(f"{p:.6g}"),
                "decision": "FE" if p < 0.05 else "RE"
            })
    return pd.DataFrame(rows)

def print_block(d: pd.DataFrame, title: str):
    print(f"\n{title}")
    print("| Outcome | Lag | chi2 | df | p-value | Decision |")
    print("|---|---:|---:|---:|---:|---|")
    for _, r in d.iterrows():
        c = "" if pd.isna(r["chi2"]) else f"{r['chi2']:.4f}"
        p = "" if pd.isna(r["p"]) else f"{float(r['p']):.4g}"
        print(f"| {r['outcome']} | {int(r['lag'])} | {c} | {int(r['df'])} | {p} | {r['decision']} |")

# ---------- main ----------
df0 = load_csv_any_encoding(CSV_FILE)
labels = resolve_columns(df0)

df = df0[[labels[k] for k in ["firm","year","sector","roa","niat","esg","emp","debt"]]]\
        .dropna(subset=[labels["firm"], labels["year"]]).copy()
df[labels["year"]] = df[labels["year"]].astype(int)
df = df.sort_values([labels["firm"], labels["year"]]).set_index([labels["firm"], labels["year"]])

# keep firms with ≥3 years
ny = df.groupby(level=0).size()
df = df[df.index.get_level_values(0).isin(ny[ny >= 3].index)].copy()

# samples
full = df.copy()
hotels = df[df[labels["sector"]].astype(str).str.contains(r"\bhotel(s)?\b", case=False, na=False)].copy()

out = pd.concat([run_block(full, labels, "full"),
                 run_block(hotels, labels, "hotels")], ignore_index=True)
out.to_csv("appendix_A1_hausman.csv", index=False)

print_block(out[out["sample"]=="full"], "Appendix A1 — Panel A: Full hospitality sample")
print_block(out[out["sample"]=="hotels"], "Appendix A1 — Panel B: Hotels-only subsample")
print(f"\nWrote {Path('appendix_A1_hausman.csv').resolve()}")

