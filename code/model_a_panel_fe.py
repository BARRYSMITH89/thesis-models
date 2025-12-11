# Model A — Panel FE (entity + year)
# Uses your columns exactly:
# Company_Code, Year, ESG_Score, Employees, Assets, Debt, ROA, NIAT, Sector

from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS

DATA_FILES = ["Final Data Including Controls.csv",
              "Final_cleaned_and_filtered_data_April_2025.csv"]

FIRM, YEAR = "Company_Code", "Year"
ESG, EMPL, ASSETS, DEBT = "ESG_Score", "Employees", "Assets", "Debt"
SECTOR = "Sector"
DEBT_RATIO = "Debt_ratio"
OUTCOMES = ["ROA", "NIAT"]
LAGS = [0, 1, 2, 3]

OUT_DIR = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "results_model_a_panel_fe.csv"

def find_csv():
    here = Path(".").resolve()
    for name in DATA_FILES:
        for p in [here / name, here / "data" / name, here.parent / name]:
            if p.exists(): return p
    raise FileNotFoundError("Data file not found.")

def read_csv_robust(p: Path):
    last = None
    for enc in ("utf-8","utf-8-sig","cp1252","latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc, low_memory=False)
            print(f"Loaded '{p.name}' with encoding '{enc}'.")
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to read CSV. Last error: {last}")

def num(df, cols):
    for c in cols: df[c] = pd.to_numeric(df[c], errors="coerce")

def build_debt_ratio(df):
    num(df, [DEBT, ASSETS])
    df[DEBT_RATIO] = df[DEBT] / df[ASSETS]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

def drop_singletons(frame):
    ent = frame.index.get_level_values(0)
    keep = ent.value_counts(); keep = keep[keep >= 2].index
    return frame.loc[ent.isin(keep)]

def fit_one(block, outcome, lag, label):
    X = block[[ESG, EMPL, DEBT_RATIO]].copy()
    y = block[outcome].copy()
    if lag: X[ESG] = X.groupby(level=0)[ESG].shift(lag)
    data = pd.concat([y, X], axis=1).dropna()
    data = drop_singletons(data)
    if data.empty or data.index.get_level_values(0).value_counts().min() < 2:
        return {"Subset": label,"Outcome": outcome,"Lag": lag,
                "N (obs)": 0,"R2_within": np.nan,"coef_ESG": np.nan,
                "p_ESG": np.nan,"status":"no_estimable_panel"}
    mod = PanelOLS(data[outcome], data[[ESG, EMPL, DEBT_RATIO]],
                   entity_effects=True, time_effects=True)
    clusters = pd.Series(data.index.get_level_values(0), index=data.index, name="entity")
    res = mod.fit(cov_type="clustered", clusters=clusters)
    return {"Subset": label,"Outcome": outcome,"Lag": lag,
            "N (obs)": int(res.nobs),
            "R2_within": float(res.rsquared_within),
            "coef_ESG": float(res.params[ESG]),
            "p_ESG": float(res.pvalues[ESG]),
            "status":"ok"}

if __name__ == "__main__":
    path = find_csv()
    df = read_csv_robust(path)
    required = [FIRM, YEAR, ESG, EMPL, ASSETS, DEBT, SECTOR, *OUTCOMES]
    miss = [c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Missing required columns: {miss}")
    num(df, [YEAR, ESG, EMPL, ASSETS, DEBT, *OUTCOMES])
    build_debt_ratio(df)
    df = df.set_index([FIRM, YEAR]).sort_index()

    results = []

    # Full sample
    for oc in OUTCOMES:
        for lg in LAGS:
            results.append(fit_one(df, oc, lg, "Full Period"))

    # COVID vs Non-COVID
yrs = df.index.get_level_values(1)
covid = df.loc[(yrs >= 2020) & (yrs <= 2021)]
non_covid = df.loc[(yrs < 2020) | (yrs >= 2022)]

for label, sub in (("COVID", covid), ("Non-COVID", non_covid)):
    for y in OUTCOMES:
        for L in LAGS:
            results.append(fit_one(sub, y, L, label))


    # Hotels Only (Sector contains 'hotel')
    hotels_mask = df.reset_index().set_index([FIRM, YEAR])[SECTOR].str.contains("hotel", case=False, na=False)
    hotels = df.loc[hotels_mask]
    for oc in OUTCOMES:
        for lg in LAGS:
            results.append(fit_one(hotels, oc, lg, "Hotels Only"))

    out = pd.DataFrame(results)[
        ["Subset","Outcome","Lag","N (obs)","R2_within","coef_ESG","p_ESG","status"]
    ]
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved results to {OUT_PATH}")

