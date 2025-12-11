# ==========================================================
# Model B — Panel Fixed Effects (entity + year), firm-clustered SEs
# Dataset columns used exactly as in your file:
#   Company_Code, Year, Environmental_Score, Social_Score, Governance_Score,
#   Employees, Assets, Debt, ROA, NIAT, Sector
#
# Produces one CSV with rows for:
#   Subset ∈ {Full Period, COVID, Non-COVID, Hotels Only}
#   Outcome ∈ {ROA, NIAT}
#   Lag ∈ {0,1,2,3}  (same lag applied to E,S,G)
#
# Columns include N, R2_within, and coef/p for E, S, G.
# ==========================================================

from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

# ---------- config ----------
DATA_FILES = [
    "Final Data Including Controls.csv",
    "Final_cleaned_and_filtered_data_April_2025.csv",
]

FIRM = "Company_Code"
YEAR = "Year"
E = "Environmental_Score"
S = "Social_Score"
G = "Governance_Score"
EMPL = "Employees"
ASSETS = "Assets"
DEBT = "Debt"
SECTOR = "Sector"

DEBT_RATIO = "Debt_ratio"
OUTCOMES = ["ROA", "NIAT"]
LAGS = [0, 1, 2, 3]

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "results_model_b_panel_fe.csv"

# ---------- io helpers ----------
def find_csv() -> Path:
    here = Path(".").resolve()
    for name in DATA_FILES:
        for p in (here / name, here / "data" / name, here.parent / name):
            if p.exists():
                return p
    raise FileNotFoundError(f"Data file not found. Looked for: {', '.join(DATA_FILES)}")

def read_csv_robust(p: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc, low_memory=False)
            print(f"Loaded '{p.name}' with encoding '{enc}'.")
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to read CSV. Last error: {last}")

# ---------- prep ----------
def coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def build_debt_ratio(df):
    coerce_numeric(df, [DEBT, ASSETS])
    df[DEBT_RATIO] = df[DEBT] / df[ASSETS]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

def drop_singletons(frame: pd.DataFrame) -> pd.DataFrame:
    ent = frame.index.get_level_values(0)
    keep = ent.value_counts()
    keep = keep[keep >= 2].index
    return frame.loc[ent.isin(keep)]

# ---------- estimation ----------
def fit_one(block, outcome, lag, label):
    # build regressors with common lag L on E,S,G within firm
    X = block[[E, S, G, EMPL, DEBT_RATIO]].copy()
    if lag:
        for c in (E, S, G):
            X[c] = X.groupby(level=0)[c].shift(lag)

    data = pd.concat([block[outcome], X], axis=1).dropna()
    data = drop_singletons(data)

    if data.empty or data.index.get_level_values(0).value_counts().min() < 2:
        return {
            "Subset": label, "Outcome": outcome, "Lag": lag,
            "N (obs)": 0, "R2_within": np.nan,
            "coef_E": np.nan, "p_E": np.nan,
            "coef_S": np.nan, "p_S": np.nan,
            "coef_G": np.nan, "p_G": np.nan,
            "status": "no_estimable_panel",
        }

    mod = PanelOLS(
        dependent=data[outcome],
        exog=data[[E, S, G, EMPL, DEBT_RATIO]],
        entity_effects=True,
        time_effects=True,
    )

    # firm clusters aligned to the model index
    clusters = pd.Series(data.index.get_level_values(0), index=data.index, name="entity")
    res = mod.fit(cov_type="clustered", clusters=clusters)

    return {
        "Subset": label, "Outcome": outcome, "Lag": lag,
        "N (obs)": int(res.nobs),
        "R2_within": float(res.rsquared_within),
        "coef_E": float(res.params.get(E, np.nan)),
        "p_E": float(res.pvalues.get(E, np.nan)),
        "coef_S": float(res.params.get(S, np.nan)),
        "p_S": float(res.pvalues.get(S, np.nan)),
        "coef_G": float(res.params.get(G, np.nan)),
        "p_G": float(res.pvalues.get(G, np.nan)),
        "status": "ok",
    }

# ---------- main ----------
if __name__ == "__main__":
    path = find_csv()
    df = read_csv_robust(path)

    required = [FIRM, YEAR, E, S, G, EMPL, ASSETS, DEBT, SECTOR, *OUTCOMES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    coerce_numeric(df, [YEAR, E, S, G, EMPL, ASSETS, DEBT, *OUTCOMES])
    build_debt_ratio(df)

    # set two-way index
    df = df.set_index([FIRM, YEAR]).sort_index()

    results = []

    # Subsets: define once, loop once
yrs = df.index.get_level_values(1)
covid = df.loc[(yrs >= 2020) & (yrs <= 2021)]
non_covid = df.loc[(yrs < 2020) | (yrs >= 2022)]
hotels = df[df[SECTOR].astype(str).str.contains("hotel", case=False, na=False)]

subsets = [
    ("Full Period", df),
    ("COVID", covid),
    ("Non-COVID", non_covid),
    ("Hotels Only", hotels),
]

for label, sub in subsets:
    for y in OUTCOMES:
        for L in LAGS:
            results.append(fit_one(sub, y, L, label))


    out = pd.DataFrame(results)[
        ["Subset","Outcome","Lag","N (obs)","R2_within",
         "coef_E","p_E","coef_S","p_S","coef_G","p_G","status"]
    ]
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved results to {OUT_PATH}")

