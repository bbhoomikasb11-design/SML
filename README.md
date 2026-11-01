# Concrete Strength Prediction Using Polynomial Regression

**Bhoomika S** · Artificial Intelligence and Data Science<br>
**Don Bosco Institute of Technology, Bengaluru**<br>
Statistical Machine Learning · Module 4 · **SDG 9 — Industry, Innovation and Infrastructure**

**ConcreteIQ** is a complete Streamlit application for predicting concrete compressive
strength, comparing linear and polynomial regression, investigating water–cement
ratio and curing age, and generating academic reports from measured experiments.

Concrete strength depends nonlinearly on composition and age. This project tests
whether polynomial features improve prediction over an additive linear baseline,
while examining validation error, residual behavior and predictor dependence.
**Academic demonstration only:** predictions are not certified structural assessments
and do not replace laboratory testing or engineering approval.

## Installation and execution

Python **3.11** is recommended. No API keys, paid services or GPU are required.

```bash
git clone https://github.com/bbhoomikasb11-design/SML.git
cd SML
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\activate`.
Open **http://localhost:8501**. The original data and trained pipelines are bundled.
If the experiment bundle is absent, the first launch trains automatically.
Restart Streamlit after changing dependencies or replacing saved model files.

`requirements.txt` pins direct and compatibility-sensitive dependencies.
`requirements-lock.txt` records the complete tested environment; install it instead
for an exact dependency replay on a compatible Python platform. Report generation
needs neither Word nor LibreOffice. Optional web fonts fall back to system fonts offline.

## Technology stack

Python, Pandas, NumPy, scikit-learn, SciPy, Statsmodels, Plotly, Matplotlib, Seaborn,
Streamlit, custom CSS, ReportLab, python-docx, joblib and pytest. Development is compatible
with VS Code and Git/GitHub. An optional notebook reuses the same project modules;
install Jupyter separately to open it.

## Eight application sections

1. **Overview Dashboard:** project context, dataset KPIs, selected model, actual vs
   predicted test observations and CV comparison.
2. **Dataset Explorer:** original records, interactive age/strength filters, summary
   statistics, schema, missing values, duplicates, outliers and data downloads.
3. **Exploratory Data Analysis:** strength distribution, all input histograms,
   correlation heatmap, selectable feature scatters, box plots and scatter matrix.
4. **Model Training:** compare all candidates with 3, 5 or 10 folds, inspect a degree,
   view train/test/CV metrics, fold scores, absolute errors and equations.
5. **Model Diagnostics:** four residual plots, robust Breusch–Pagan and Shapiro–Wilk
   tests, unusual errors, original-input VIF and expanded-design conditioning.
6. **Feature Analysis:** observed ratio/age trends, Pearson and Spearman associations,
   age-group summaries and fixed-mixture model scenarios.
7. **Strength Prediction:** eight validated inputs, real training-data presets, reset,
   extrapolation warnings, session history and CSV download.
8. **Project Report:** generate matching PDF and editable Word reports for the active run.

The responsive interface uses reusable chart styling and a navy/blue/teal palette.
Session retraining does not overwrite the bundled baseline models. Prediction history
stays in the browser session; reports are organized by experiment ID.

## Dataset and provenance

[UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength),
I-Cheng Yeh (2007), DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67), **CC BY 4.0**.

Inputs: cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate
and fine aggregate in **kg/m³**, plus age in **days**. Target: strength in **MPa**.

- Original data: **1,030 × 9**, with **0 missing cells** and **0 invalid rows**.
- The modeling copy removes **25 exact duplicates**, leaving **1,005 observations**.
- There are **996 distinct input vectors** after deduplication. Nine remaining input
  repetitions with different strengths are retained and grouped during splitting.
- The original workbook and archive remain unchanged. Source metadata and a SHA-256
  checksum are recorded in [provenance.json](data/raw/provenance.json).
- Checks cover schema, numeric conversion, finiteness, non-negative masses, positive
  cement/water/age/strength, and integer ages. Missing targets would be excluded;
  predictor imputation happens inside training folds only.
- IQR outliers are **flagged, not automatically deleted**.

Offline fallback: bundled workbook, then cached ZIP. If both are absent, the loader
tries the UCI archive and legacy UCI XLS endpoint. Failure is explicit; it never
fabricates observations. See [data/README.md](data/README.md).

## Model methodology

1. Prespecify eight inputs and polynomial degrees **1, 2 and 3**.
2. Seeded **80/20 split of unique input-vector groups**, seed 42: **803 training rows,
   202 test rows**. Identical inputs cannot cross the boundary.
3. **Shuffled five-fold KFold on training groups**, mapped back to row indices.
   Fold row counts can differ. Every candidate uses identical splits.
4. Each fold fits: median imputation → input standardization →
   `PolynomialFeatures(include_bias=False)` → term standardization → `LinearRegression`.
5. Select **lowest mean training CV RMSE**, using fewer terms to break ties. Test
   scores never determine the degree.
6. Refit candidates on all training rows and evaluate the same held-out observations.
   The saved pipeline remains the evaluated training-fit model; there is no silent
   full-data refit.

```text
Linear:     y_hat = beta_0 + sum_j beta_j x_j
Polynomial: y_hat = beta_0 + sum_(1 <= |alpha| <= d) beta_alpha product_j z_j^alpha_j
            z_j = (x_j - training_mean_j) / training_sd_j
```

Polynomial regression is nonlinear in inputs but linear in coefficients. All powers
and interactions are included: **8, 44 and 164 terms**, excluding intercept. Expanded
terms are standardized too. Scaling improves numerical conditioning but does not
remove multicollinearity. Water/cement is an analysis variable, not a ninth predictor.

## Measured baseline results

Experiment **9ef455423207**, rounded from
[model_comparison.csv](artifacts/metrics/model_comparison.csv).

| Model | Terms | Train RMSE | Test MAE | Test MSE | Test RMSE | Test R² | CV RMSE ± SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | 8 | 10.2175 | 8.3617 | 109.4492 | 10.4618 | 0.5312 | 10.3676 ± 0.4611 |
| Polynomial 2 | 44 | 7.1810 | 5.7520 | 56.0542 | 7.4869 | 0.7599 | 7.8328 ± 0.4690 |
| Polynomial 3 | 164 | 4.3779 | 4.5353 | 38.1475 | 6.1764 | 0.8366 | 7.4893 ± 1.7262 |

MAE/RMSE are MPa, MSE is MPa² and R² is unitless. CV RMSE is the unweighted mean
of fold RMSEs, not pooled out-of-fold RMSE. SD is sample fold standard deviation,
**not a confidence interval**. Degree 3 is selected, but its wider fold variation and
training-validation gap matter. Degree 2 remains a useful simpler comparator.
The experiment does not establish statistically significant superiority of degree 3.

MAE = mean absolute error; MSE = mean squared error; RMSE = sqrt(MSE);
R² = 1 − SSE/SST, with SST centered on the evaluation partition's actual target mean.
All metrics displayed in the application and reports are computed, never hardcoded.

## Statistical and feature interpretation

- Residual = **actual − predicted**, on held-out observations. Positive means underprediction.
- Robust Breusch–Pagan/Koenker LM **p = 0.07741**: no rejection at 5%, **not proof of
  homoscedasticity**. The auxiliary design contains an intercept and eight original
  test inputs. This is an exploratory holdout diagnostic, not classical in-sample inference.
- Shapiro–Wilk **p ≈ 1.043 × 10⁻⁶** rejects normality. Normal errors are not required
  for point prediction. No normal-theory individual prediction intervals are claimed.
- VIF uses **training data with an intercept**; the intercept's VIF is omitted. Cement,
  slag, fly ash, water and fine aggregate exceed 5. Predictors are retained with an
  explicit caution about coefficient interpretation.
- The selected expanded design has rank **165/165** and condition number **2.72 × 10³**.
  Original-input VIF does not describe every expanded polynomial dependence.
- Water/cement–strength Pearson **r = −0.4894**; age–strength Pearson **r = 0.3374**,
  Spearman **rho = 0.6054**. These are unadjusted observations, not causal effects.
- Descriptive LOWESS uses a 60% neighborhood for age, spanning its dense tied 28-day
  block, and 35% for ratio. Scenario curves instead use the trained model.

The split does not group the same formulation across different ages. Unknown source
or batch dependence and absent curing conditions limit generalization. Combinations
inside marginal ranges can still be unusual. Polynomial predictions can be negative
or non-monotonic; these are disclosed instead of silently clipped. Test RMSE is an
aggregate error measure, not an interval for one prediction.

## Reproduction and reports

```bash
python -m src.reproduce             # rebuild the full baseline and both reports
pytest -q
```

Individual stages:

```bash
python -m src.data_loader
python -m src.model_training
python -m src.export_figures
python -m reports.generate_report
```

`python -m src.model_training --folds 3` (or `10`) explicitly replaces the disk
experiment. `python -m src.reproduce` restores the documented five-fold baseline.
Restart Streamlit after replacing disk models because its initial bundle is cached.
Only load trusted `.joblib` artifacts: pickle-based formats can execute code. The app
checks the scikit-learn version and source checksum; rebuild if either changes.

Baseline academic deliverables, **22 pages each**, with contents, numbered pages,
11 numbered figures, 10 numbered tables, equations, references and measured results:

- [PDF report](reports/output/9ef455423207/concrete_strength_report.pdf)
- [Editable DOCX report](reports/output/9ef455423207/concrete_strength_report.docx)
- [Result manifest](reports/output/9ef455423207/report_manifest.json)

Both reports share one content tree and cover all requested academic sections. The
DOCX contents is static for the generated edition; edits can change pagination.
No institutional certificate/declaration/acknowledgement templates were supplied, so
those optional pages are omitted. Rendering utilities are QA-only dependencies;
users only need `requirements.txt` to generate reports.

## Project structure

```text
app.py                         Streamlit entry point
config.py                      Schema, units and defaults
components/                    Sidebar, styling, charts and eight page implementations
src/data_loader.py             Acquisition, provenance and audit
src/preprocessing.py           Fold-local pipelines and grouped splitting
src/eda.py                     Interactive exploratory charts
src/feature_engineering.py     Ratio and age groups for analysis
src/model_training.py          Training, CV, selection and persistence
src/evaluation.py              MAE, MSE, RMSE and R²
src/diagnostics.py             Residual tests, VIF and design diagnostics
src/feature_analysis.py        Associations and descriptive trends
src/prediction.py              Input validation and extrapolation warnings
src/export_figures.py          Publication figures
src/reproduce.py               Full baseline reproduction
models/                        Experiment bundle and selected Pipeline
data/raw/                      Original workbook, ZIP, readme and provenance
data/processed/                Modeling copy and split assignments
artifacts/                     Figures, metrics, predictions and model metadata
artifacts/interactive/         Offline interactive EDA HTML with a shared Plotly JS asset
reports/                       Shared report content and PDF/DOCX generator
reports/output/                Reports and manifests by experiment ID
notebooks/                     Optional EDA notebook using shared modules
tests/                         Data, model, diagnostics, prediction, reports and UI tests
.github/workflows/tests.yml    Python 3.11 CI validation
```

## Testing and completed phases

The suite checks dataset loading and invalid values; train/test and fold isolation;
fold-local preprocessing and independently recomputed CV scores; metric consistency;
reproducibility and serialized model reload; invalid inputs and extrapolation;
VIF and finite trends; generated report metrics and page mapping; every navigation
section; presets, reset, history, retraining, report controls and empty filters.
See [VALIDATION.md](VALIDATION.md) for executed results and visual QA.

The ten implementation phases are complete: architecture, original data audit,
preprocessing/EDA, regression pipelines, evaluation/CV, diagnostics, ratio/age analysis,
Streamlit interface, academic reports, and testing/documentation. Actual execution
results informed each stage. The notebook is supplementary; the application and
reports use the shared implementation in `src/`.
