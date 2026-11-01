# Dataset provenance

Source: [UCI Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).

I-Cheng Yeh (2007), DOI [10.24432/C5PK67](https://doi.org/10.24432/C5PK67).
License: **CC BY 4.0**. Please retain this attribution when redistributing the data.

`raw/concrete_original.zip`, `raw/Concrete_Data.xls` and `raw/Concrete_Readme.txt`
are the original source files. `raw/provenance.json` records the source, checksum
and audit. No observations were fabricated or overwritten.

`processed/concrete_clean.csv` renames the nine columns, retains original zero-based
row indices, and removes 25 exact duplicate records. This leaves 1,005 observations.
Nine remaining repeated input vectors have different targets. They remain together
when splitting training/test and cross-validation folds. Missing targets would be
excluded; predictor imputation stays inside the model pipeline.

`processed/split_assignment.csv` identifies each baseline row's train/test partition.
The experiment bundle also includes the exact CV indices. All 1,030 original rows
remain available in the Dataset Explorer. No IQR-flagged valid record is deleted.

Offline fallback: the original workbook is bundled. If it is missing, ingestion
extracts the cached archive. If both are missing it attempts the UCI archive and
then the legacy UCI workbook endpoint. It fails explicitly if neither succeeds.
