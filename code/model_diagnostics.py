"""
model_diagnostics.py
Fixed-Effects diagnostics for ROA and NIAT on the full hospitality panel.

Data:
  ~/thesis_models/data/Final Data Including Controls.csv
Identifiers:
  entity = Company_Code, time = Year
Predictors:
  ESG_Score, Employees, Debt, Assets
Output:
  ~/thesis_models/outputs/model_diagnostics.csv
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import PanelOLS


# ---------- 1) Load data ----------
DATA_PATH = os.path.expanduser(
    "~/thesis_models/data/Final Data Including Controls.csv"
)

def read_csv_robust(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")

df = read_csv_robust(DATA_PATH)

# required columns
ID_COL, TIME_COL = "Company_Code", "Year"
PREDICTORS = ["ESG_Score", "Employees", "Debt", "Assets"]
DEPENDENTS = ["ROA", "NIAT"]

required = set([ID_COL, TIME_COL] + PREDICTORS + DEPENDENTS)
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# panel index
df = df.set_index([ID_COL, TIME_COL]).sort_index()

def add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


# ---------- 2) FE models + diagnostics ----------
rows = []

for dep in DEPENDENTS:
    used = PREDICTORS + [dep]
    d = df[used].dropna()

    y = d[dep]
    X = add_const(d[PREDICTORS])

    # FE with entity and time effects; two-way clustered SEs
    fe = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = fe.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    # core fit stats
    r2_w = res.rsquared_within
    r2_b = res.rsquared_between
    r2_o = res.rsquared_overall
    f_stat = getattr(res.f_statistic, "stat", np.nan)
    f_pval = getattr(res.f_statistic, "pval", np.nan)

    # residuals (MultiIndex: entity,time)
    e = res.resids.dropna().sort_index()

    # ---- Wooldridge AR(1) (pooled) ---------------------------------------
    e_lag = e.groupby(level=0).shift(1)
    mask = e.notna() & e_lag.notna()
    if mask.any():
        rho, _ = stats.pearsonr(e[mask], e_lag[mask])
        n_eff = int(mask.sum())
        wool_stat = n_eff * (rho ** 2)
        wool_pval = 1 - stats.chi2.cdf(wool_stat, 1)
    else:
        wool_stat, wool_pval = np.nan, np.nan

    # ---- Pesaran CD (two-way de-mean) ------------------------------------
    # matrix with rows=time, cols=entities
    M = e.unstack(level=0)  # index=time, columns=entity
    M = M.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if M.shape[0] >= 3 and M.shape[1] >= 2:
        # de-mean by entity and time
        M = M - M.mean(axis=0)
        M = M.sub(M.mean(axis=1), axis=0)
        R = M.corr()
        N = R.shape[0]
        iu = np.triu_indices(N, k=1)
        rbar = np.nanmean(R.values[iu])
        T = M.shape[0]
        cd_stat = np.sqrt(T) * rbar * np.sqrt(N * (N - 1) / 2.0)
        cd_pval = 2 * (1 - stats.norm.cdf(abs(cd_stat)))
    else:
        cd_stat, cd_pval = np.nan, np.nan

    # ---- Groupwise heteroskedasticity proxy ------------------------------
    gv = e.groupby(level=0).var()
    if gv.size >= 3 and gv.mean() > 0:
        wald_stat = (gv.var() / (gv.mean() ** 2)) * gv.size
        wald_pval = 1 - stats.chi2.cdf(wald_stat, 1)
    else:
        wald_stat, wald_pval = np.nan, np.nan

    rows.append({
        "Dependent_Var": dep,
        "Model": "FE (entity + time)",
        "Obs_used": int(d.shape[0]),
        "Entities": int(d.index.get_level_values(0).nunique()),
        "Years": int(d.index.get_level_values(1).nunique()),
        "R2_within": r2_w,
        "R2_between": r2_b,
        "R2_overall": r2_o,
        "F_stat": f_stat,
        "F_pval": f_pval,
        "Wooldridge_stat": wool_stat,
        "Wooldridge_pval": wool_pval,
        "PesaranCD_stat": cd_stat,
        "PesaranCD_pval": cd_pval,
        "Wald_stat": wald_stat,
        "Wald_pval": wald_pval
    })

# ---------- 3) Save ----------
OUT_PATH = os.path.expanduser(
    "~/thesis_models/outputs/model_diagnostics.csv"
)
pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
print(f"Diagnostics saved to {OUT_PATH}")

