# Validation record

Executed locally on **1 November 2026**, Python **3.11.15**, macOS arm64.
Baseline experiment: **9ef455423207**, seed **42**, **five-fold grouped CV**.

## Automated results

`pytest -q --junitxml=artifacts/metrics/pytest-results.xml`: **32 passed**, no warnings.
The saved text and JUnit files are in `artifacts/metrics/`.
`pip check`: **No broken requirements found**.

Tests verify dataset dimensions, schema, physical constraints, missing-value handling,
deduplication, train/test and fold group isolation, validation coverage, fold-local
preprocessing, independently recomputed CV scores, metric calculations, deterministic
retraining, serialized pipeline reload, valid and invalid predictions, extrapolation
warnings, residual definitions, intercept-aware VIF, finite descriptive curves,
PDF/DOCX generation, numerical consistency, report contents/page mapping, all eight
navigation pages, input presets/reset, history, retraining, report controls and empty
filter states. Runtime warnings fail the suite.

## Executed workflow

- Original UCI archive downloaded, unmodified workbook preserved, checksum recorded.
- `python -m src.reproduce` rebuilt the baseline model bundle, figures, diagnostic
  exports and both reports from actual data.
- Saved model selection: Polynomial 3. Test RMSE 6.1764 MPa; test R² 0.8366.
- The same train/test and CV partitions are used for every candidate; no identical
  eight-input vector crosses a boundary.
- Tests also exercised a three-fold session experiment without replacing baseline models.

## Interface checks

- All eight pages executed with Streamlit AppTest without errors.
- Browser visual review at desktop size and **390 × 844** responsive viewport.
- Mobile navigation drawer opens/closes; cards stack; chart components resize.
- Live prediction using the original training preset produced **67.69 MPa** for
  cement 540, slag 0, fly ash 0, water 162, superplasticizer 2.5, coarse aggregate 1040,
  fine aggregate 676 kg/m³, and age 28 days. This is a model estimate, not a measured
  or certified strength claim. Water/cement displayed as 0.300.
- Session history appeared and the CSV download completed in the browser.
- PDF report download completed in the browser; PDF/DOCX controls and generation
  were additionally exercised in automated tests.
- Initial Arrow/NumPy incompatibility was fixed by pinning Arrow 19.0.1 and restarting
  the server. Preset callback state handling was corrected after a regression test.
- Age LOWESS was adjusted to span the dense tied 28-day block, eliminating a numerical
  warning. Model selection and regression results were unchanged.

## Report checks

- PDF and editable DOCX each render to **22 pages**.
- All numbered contents entries map to the correct rendered page in both formats.
- Both reports contain **11 figures** and **10 tables** from the shared content tree.
- Every page was visually reviewed for clipping, table overflow, figure labels,
  missing glyphs, page numbers and heading placement. The DOCX cover's inherited
  decorative rule was removed.
- Report metrics are checked against the experiment bundle and manifest.
- PDF generated with ReportLab; DOCX with python-docx. DOCX layout checked using the
  document skill's renderer with an isolated official LibreOffice 26.8.2 download;
  Poppler rendered page images. These QA dependencies are not required to run the app.

## Scope of assurance

This validates an academic local application, not a structural engineering product.
Statistical significance of the selected degree, causal effects, calibrated individual
prediction intervals, source-level independence and external-site generalization are
not established. The README, dashboard and reports explain these limitations.
The GitHub Actions workflow is included; this record describes local execution and
must not be read as a claim that hosted CI has already run successfully.
