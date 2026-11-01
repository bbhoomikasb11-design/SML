"""Acquire and audit the original UCI workbook, never synthetic fallback rows."""
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from config import DATA_PAGE, DATA_URL, FEATURES, ROOT, TARGET

RAW = ROOT / "data/raw"
WORKBOOK = RAW / "Concrete_Data.xls"


def acquire():
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / "concrete_original.zip"
    if WORKBOOK.exists():
        return WORKBOOK
    if not archive.exists():
        errors = []
        for url in [DATA_URL, "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"]:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                if url.endswith(".xls"):
                    pd.read_excel(io.BytesIO(response.content))  # reject HTML/error bodies
                    WORKBOOK.write_bytes(response.content)
                    return WORKBOOK
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    if not any(n.endswith("Concrete_Data.xls") for n in z.namelist()):
                        raise ValueError("UCI archive has no expected workbook")
                archive.write_bytes(response.content)
                break
            except (requests.RequestException, ValueError, zipfile.BadZipFile) as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError("UCI download failed. Restore the bundled data/raw/Concrete_Data.xls. " + "; ".join(errors))
    with zipfile.ZipFile(archive) as z:
        for name in ["Concrete_Data.xls", "Concrete_Readme.txt"]:
            match = next((n for n in z.namelist() if n.endswith(name)), None)
            if match:
                (RAW / name).write_bytes(z.read(match))
    return WORKBOOK


def load_data():
    path = acquire()
    frame = pd.read_excel(path)
    expected = ["cement", "blast furnace slag", "fly ash", "water", "superplasticizer",
                "coarse aggregate", "fine aggregate", "age", "concrete compressive strength"]
    normalized = [" ".join(str(c).lower().split()) for c in frame.columns]
    if len(normalized) != 9 or any(not name.startswith(prefix) for name, prefix in zip(normalized, expected)):
        raise ValueError(f"Unexpected UCI schema: {list(frame.columns)}")
    frame.columns = FEATURES + [TARGET]
    return frame.apply(pd.to_numeric, errors="raise").astype(float)


def audit_data(frame):
    if list(frame.columns) != FEATURES + [TARGET]:
        raise ValueError("Dataset schema does not match the nine expected numeric columns")
    values = frame.to_numpy(dtype=float)
    invalid = (~np.isfinite(values) & ~np.isnan(values)).any(axis=1) | (values < 0).any(axis=1)
    invalid |= (frame[["Cement", "Water", "Age", TARGET]] <= 0).any(axis=1).to_numpy()
    invalid |= (frame.Age.dropna() % 1 != 0).reindex(frame.index, fill_value=False).to_numpy()
    q1, q3 = frame.quantile(.25), frame.quantile(.75)
    iqr = q3 - q1
    outliers = ((frame < q1 - 1.5 * iqr) | (frame > q3 + 1.5 * iqr)).sum()
    return {"rows": len(frame), "columns": len(frame.columns), "features": len(FEATURES),
            "missing": frame.isna().sum().to_dict(), "duplicate_rows": int(frame.duplicated().sum()),
            "repeated_inputs": int(frame.duplicated(FEATURES).sum()),
            "invalid_rows": int(invalid.sum()), "iqr_flags": outliers.to_dict()}


def prepare_data(frame):
    audit = audit_data(frame)
    if audit["invalid_rows"]:
        raise ValueError("Invalid values found; investigate original observations before training")
    # Missing outcomes cannot be learned. Predictor imputation is fold-local.
    return frame.dropna(subset=[TARGET]).drop_duplicates().copy()


def save_dataset_audit(frame):
    audit = audit_data(frame)
    digest = hashlib.sha256(WORKBOOK.read_bytes()).hexdigest()
    metadata = {"source": DATA_PAGE, "download_url": DATA_URL, "sha256": digest,
                "audit_utc": datetime.now(timezone.utc).isoformat(), "license": "CC BY 4.0",
                "attribution": "I-Cheng Yeh. Concrete Compressive Strength. UCI Machine Learning Repository. DOI: 10.24432/C5PK67",
                "audit": audit}
    (RAW / "provenance.json").write_text(json.dumps(metadata, indent=2))
    processed = ROOT / "data/processed"
    processed.mkdir(exist_ok=True)
    prepare_data(frame).to_csv(processed / "concrete_clean.csv", index_label="source_row")
    return metadata


if __name__ == "__main__":
    print(json.dumps(save_dataset_audit(load_data()), indent=2))
