#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/Final Data Including Controls.csv")
OUT  = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, encoding="ISO-8859-1")

# --- column detection ---
firm_col = "Company_Code" if "Company_Code" in df.columns else ("Company" if "Company" in df.columns else None)
year_col = "Year"
sector_col = "Sector" if "Sector" in df.columns else None

if firm_col is None or year_col not in df.columns:
    raise ValueError("Expected columns not found. Need 'Company_Code' or 'Company' and 'Year'.")

# helper to label panels
def is_hotels(x):
    if sector_col is None:
        return pd.Series(False, index=x.index)
    return x[sector_col].astype(str).str.contains("hotel", case=False, na=False)

df["panel"] = "Full sample"
df.loc[is_hotels(df), "panel"] = "Hotels only"
df["period"] = np.where(df[year_col].between(2020, 2021), "COVID 2020–2021", "Non-COVID 2008–2019 & 2022–2024")

# --- sample overview by panel/period ---
def overview(sub, label):
    entities = sub[firm_col].nunique()
    years_min, years_max = sub[year_col].min(), sub[year_col].max()
    obs = len(sub)
    return pd.Series({
        "label": label,
        "entities": entities,
        "obs": obs,
        "years_min": years_min,
        "years_max": years_max,
        "years_span": int(years_max - years_min + 1)
    })

rows = []
rows.append(overview(df, "Full sample"))
rows.append(overview(df[df["panel"]=="Hotels only"], "Hotels only"))
rows.append(overview(df[df["period"]=="COVID 2020–2021"], "COVID 2020–2021"))
rows.append(overview(df[df["period"]!="COVID 2020–2021"], "Non-COVID 2008–2019 & 2022–2024"))
pd.DataFrame(rows).to_csv(OUT/"sample_overview.csv", index=False)

# --- sector breakdown (optional, if Sector exists) ---
if sector_col:
    b = (df
         .groupby(sector_col, dropna=False)
         .agg(entities=(firm_col, "nunique"),
              obs=("Year","size"))
         .sort_values("obs", ascending=False)
         .reset_index())
    b.to_csv(OUT/"sector_breakdown.csv", index=False)

# --- descriptive stats for key vars by panel/period split ---
vars_keep = [c for c in ["ESG_Score","ROA","NIAT","Employees","Debt"] if c in df.columns]
if not vars_keep:
    raise ValueError("None of ESG_Score, ROA, NIAT, Employees, Debt found in data.")

def desc_table(sub):
    stats = {}
    for v in vars_keep:
        s = sub[v].astype("float64")
        stats[v+"_count"] = s.count()
        stats[v+"_mean"]  = s.mean()
        stats[v+"_sd"]    = s.std(ddof=1)
        stats[v+"_min"]   = s.min()
        stats[v+"_max"]   = s.max()
    return pd.Series(stats)

tables = {
    "desc_full.csv": df,
    "desc_hotels.csv": df[df["panel"]=="Hotels only"],
    "desc_covid.csv": df[df["period"]=="COVID 2020–2021"],
    "desc_noncovid.csv": df[df["period"]!="COVID 2020–2021"],
}

for name, sub in tables.items():
    row = desc_table(sub)
    # add high-level counts
    row["entities"] = sub[firm_col].nunique()
    row["obs"] = len(sub)
    row["years_min"] = sub[year_col].min()
    row["years_max"] = sub[year_col].max()
    pd.DataFrame([row]).to_csv(OUT/name, index=False)
