
---

# Thesis Replication Folder

## Purpose

This folder contains the code and outputs used to run the econometric analyses in my thesis.
The underlying Refinitiv data are proprietary and therefore not included in this public repository:

“Corporate Social Responsibility and Financial Performance within the Global Hospitality Industry and Hotel Sector”.

Everything is designed so that an external reviewer can rebuild the Python environment and rerun all scripts from scratch.

## Folder structure

* `code/`
  All Python scripts used to generate descriptive statistics, figures, model estimates and diagnostics.

* `data/`
  Contains the **single** cleaned analysis dataset used in the thesis:
  `Final Data Including Controls.csv`
  (panel of hospitality-related firms with ESG scores, financial variables and controls).
  Raw Refinitiv data are not included because of licensing restrictions.

* `outputs/`
  CSV and PNG files produced by the scripts in `code/`.
  These include descriptive tables, regression results, figures and diagnostics.
  Re-running the scripts will overwrite these files with the same names.

* `requirements.txt`
  List of Python packages (and versions where relevant) required to run the scripts.

* `.venv/` (may or may not be present)
  Local virtual environment used by the author.
  This is **not** required for replication and can be ignored or deleted by other users.

## Software requirements

* Python 3.12 (or a recent 3.x version; the project was developed under Python 3.12).
* A terminal / command prompt.
* The ability to create and activate a Python virtual environment.

## Quick start (recommended sequence)

1. Open a terminal in the project root (`thesis_models`).

2. Create a fresh virtual environment (if you do not already have one):

   ```bash
   python -m venv .venv
   ```

3. Activate the environment.

   * Linux / macOS:

     ```bash
     source .venv/bin/activate
     ```

   * Windows (PowerShell):

     ```powershell
     .venv\Scripts\Activate.ps1
     ```

4. Install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Ensure you are in the project root and that the `data/` and `code/` folders are visible:

   ```bash
   ls
   # should show: code  data  outputs  requirements.txt  README.txt  ...
   ```

6. Run the scripts as needed (see the “Scripts overview” below).
   Example:

   ```bash
   python code/sample_descriptives.py
   ```

   All outputs will be written into the `outputs/` subfolders.

## Scripts overview

All scripts assume that the current working directory is the project root (`thesis_models`), **not** the `code/` folder.

## Descriptives and figures

* `code/sample_descriptives.py`
  Generates basic sample characteristics and related CSV tables (e.g. `sample_overview.csv`, `desc_full.csv`, `desc_hotels.csv`, `desc_covid.csv`, `desc_noncovid.csv`) in `outputs/descriptives/`.

* `code/describe_esg.py`
  Produces descriptives for the overall ESG score and related output files in `outputs/tables/`.

* `code/describe_pillars.py`
  Produces descriptives for the E, S and G pillar scores and related output files in `outputs/tables/`.

* `code/controls_descriptives.py`
  Creates descriptive statistics for the control variables (e.g. leverage, firm size, employees) and writes them to `outputs/tables/`.

* `code/depvar_distribution.py`
  Creates distribution plots for ROA and NIAT (e.g. histograms) and saves PNG files in `outputs/figures/` and `outputs/descriptives/`.

* `code/niat_descriptives_and_corr.py`
  Produces NIAT descriptives and correlation information (including correlation with ROA and ESG), stored in `outputs/tables/` and related figures.

* `code/correlation_matrix.py`
  Computes the main correlation matrix of key variables and exports CSV and PNG files in `outputs/tables/` and `outputs/figures/`.

## Main regression models

All model scripts use the same underlying dataset:

* Input data file (for every script below):
  `data/Final Data Including Controls.csv`

* Panel structure: firm‐ and year‐level fixed effects where relevant.

* Standard errors: clustered (entity and year) where specified.

* `code/model_a_panel_fe.py`
  Runs all **Model A** fixed-effects panel regressions for ROA and NIAT used in the main thesis tables, including full sample and sub-sample splits (e.g. COVID vs non-COVID, hotels vs full hospitality sample).
  Outputs: regression result tables in `outputs/` (e.g. `results_model_a_panel_fe.csv` and related files).

* `code/model_b_panel_fe.py`
  Runs all **Model B** regressions (extended specification with additional controls).
  Outputs: regression result tables in `outputs/` (e.g. `results_model_b_panel_fe.csv`, `model_b_components_by_period.csv`).

## Diagnostics and robustness

* `code/model_diagnostics.py`
  Produces diagnostics for the main panel models (e.g. tests for heteroskedasticity / serial correlation) and writes results to `outputs/model_diagnostics.csv`.

* `code/vif_diagnostics.py`
  Calculates variance inflation factors (VIF) for key regressors and saves summary output (e.g. `outputs/vif_summary.csv`).

* `code/hausman_appendix_A1.py`
  Runs Hausman tests comparing fixed-effects and random-effects specifications for the key models and writes results to `outputs/Appendix_A1_Hausman.csv` (and related files).

## Controls-only regression summary

* `code/controls_summary.py`
  Re-estimates the FE panel models using only the control variables (ESG removed), for:

  * full sample,
  * hotels only,
  * COVID period,
  * non-COVID period,
    and for the different lags used in the thesis.

  Outputs (all in `outputs/`):

  * `controls_full.csv`
  * `controls_hotels.csv`
  * `controls_covid.csv`
  * `controls_noncovid.csv`
  * `controls_availability.csv`
  * `variable_availability.csv`
    and related summary files used in the thesis discussion.

## Sample and appendix tables

* `code/sample_and_appendices.py`
  Produces additional tables and descriptive breakdowns used in the appendices (e.g. sector breakdowns, ESG summaries by sector and time period) and stores them in `outputs/` (including files such as `sector_breakdown.csv`, `sector_means_esg_pillars.csv`, `appendix5_desc_combined.csv`).

## Re-running everything

To regenerate all key outputs used in the thesis from a fresh clone:

```bash
# 1. From the project root:
python -m venv .venv

# 2. Activate environment
source .venv/bin/activate          # Linux / macOS
# or
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install packages
pip install -r requirements.txt

# 4. Run scripts (order suggestion)
python code/sample_descriptives.py
python code/describe_esg.py
python code/describe_pillars.py
python code/controls_descriptives.py
python code/depvar_distribution.py
python code/niat_descriptives_and_corr.py
python code/correlation_matrix.py
python code/model_a_panel_fe.py
python code/model_b_panel_fe.py
python code/model_diagnostics.py
python code/vif_diagnostics.py
python code/hausman_appendix_A1.py
python code/controls_summary.py
python code/sample_and_appendices.py
```

All tables and figures in the thesis are derived from the outputs produced by these scripts.

