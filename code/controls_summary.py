import pathlib
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Final Data Including Controls.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

ENTITY, TIME = "Company_Code", "Year"
Y_LIST = ["ROA", "NIAT"]
ESG = "ESG_Score"
CONTROLS = ["Employees", "Debt"]

def prep(df: pd.DataFrame) -> pd.DataFrame:
    # keep only needed cols
    need = ["Sector", ENTITY, TIME, ESG, *Y_LIST, *CONTROLS]
    df = df.loc[:, need].copy()

    # coerce numerics safely
    for c in [ESG, *Y_LIST, *CONTROLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalise keys
    df[ENTITY] = df[ENTITY].astype(str)
    df[TIME] = pd.to_numeric(df[TIME], errors="coerce").astype("Int64")

    # drop rows without keys
    df = df.dropna(subset=[ENTITY, TIME])
    return df.sort_values([ENTITY, TIME]).set_index([ENTITY, TIME])

def valid_panel(df_idxed: pd.DataFrame) -> bool:
    return (
        df_idxed.index.get_level_values(0).nunique() >= 2
        and df_idxed.index.get_level_values(1).nunique() >= 2
        and len(df_idxed) > 10
    )

def try_fit(df_idxed: pd.DataFrame, ycol: str, lag: int, label: str):
    row = {
        "panel": label, "outcome": ycol, "lag": lag,
        "entities": int(df_idxed.index.get_level_values(0).nunique()),
        "obs_used": int(len(df_idxed)), "clustered_SEs": "entity+year",
        "coef_Employees": float("nan"), "p_Employees": float("nan"),
        "coef_Debt": float("nan"), "p_Debt": float("nan"),
        "note": ""
    }
    try:
        exog = [f"{ESG}_lag{lag}", *CONTROLS]
        y = df_idxed[ycol]
        X = df_idxed[exog]
        mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        row.update(
            coef_Employees=res.params.get("Employees", float("nan")),
            p_Employees=res.pvalues.get("Employees", float("nan")),
            coef_Debt=res.params.get("Debt", float("nan")),
            p_Debt=res.pvalues.get("Debt", float("nan")),
        )
    except Exception as e:
        row["note"] = f"skipped: {type(e).__name__}"
    return row

def run_block(df_idxed: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for lag in [0, 1, 2]:
        df_lag = df_idxed.copy()
        df_lag[f"{ESG}_lag{lag}"] = df_lag.groupby(level=0)[ESG].shift(lag)
        cols = [*Y_LIST, f"{ESG}_lag{lag}", *CONTROLS]
        df_lag = df_lag[cols].dropna()
        if not valid_panel(df_lag):
            for y in Y_LIST:
                rows.append({
                    "panel": label, "outcome": y, "lag": lag,
                    "entities": int(df_lag.index.get_level_values(0).nunique()) if len(df_lag) else 0,
                    "obs_used": int(len(df_lag)),
                    "clustered_SEs": "entity+year",
                    "coef_Employees": float("nan"), "p_Employees": float("nan"),
                    "coef_Debt": float("nan"), "p_Debt": float("nan"),
                    "note": "skipped: insufficient panel"
                })
            continue
        for y in Y_LIST:
            rows.append(try_fit(df_lag, y, lag, label))
    return pd.DataFrame(rows)

def main():
    df = prep(pd.read_csv(DATA, encoding="ISO-8859-1"))
    # full
    full = run_block(df, "Full sample")
    full.to_csv(OUT / "controls_full.csv", index=False)

    # hotels only
    hotels_mask = df.reset_index().Sector.str.contains("hotel", case=False, na=False)
    hotels = df.reset_index().loc[hotels_mask].set_index([ENTITY, TIME])
    run_block(hotels, "Hotels only").to_csv(OUT / "controls_hotels.csv", index=False)

    # covid splits
    idx = df.reset_index()
    covid = idx[idx[TIME].between(2020, 2021)].set_index([ENTITY, TIME])
    noncovid = idx[idx[TIME].between(2008, 2019) | idx[TIME].between(2022, 2024)].set_index([ENTITY, TIME])
    run_block(covid, "COVID 2020–2021").to_csv(OUT / "controls_covid.csv", index=False)
    run_block(noncovid, "Non-COVID 2008–2019 & 2022–2024").to_csv(OUT / "controls_noncovid.csv", index=False)

if __name__ == "__main__":
    main()

