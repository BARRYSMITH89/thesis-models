from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "sample_descriptives.py",   # sample_overview.csv, desc_*.csv, sector_breakdown.csv
    "controls_summary.py",      # controls_full/covid/noncovid/hotels.csv
    "vif_diagnostics.py",       # vif_summary.csv
    "model_diagnostics.py",     # model_diagnostics.csv
    "hausman_appendix_A1.py",   # Appendix_A1_Hausman.xlsx
]

for name in SCRIPTS:
    print(f"==> Running {name}")
    runpy.run_path(str(CODE / name), run_name="__main__")

expected = [
    "sample_overview.csv",
    "sector_breakdown.csv",
    "desc_full.csv", "desc_covid.csv", "desc_noncovid.csv",
    "controls_full.csv", "controls_covid.csv", "controls_noncovid.csv", "controls_hotels.csv",
    "vif_summary.csv", "model_diagnostics.csv",
    "Appendix_A1_Hausman.xlsx",
]
missing = [f for f in expected if not (OUT / f).exists()]
print("\nCreated outputs in ./outputs/")
for f in expected:
    print(f" - {f}{'  (missing)' if f in missing else ''}")
if missing:
    raise SystemExit(f"\nSome expected files are missing: {missing}")
print("\nDone.")
