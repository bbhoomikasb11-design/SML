"""Shared schema, paths and reproducibility defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATURES = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer",
            "Coarse Aggregate", "Fine Aggregate", "Age"]
TARGET = "Strength"
LABELS = {f: f"{f} (kg/m³)" for f in FEATURES}
LABELS.update(Age="Age (days)", Strength="Compressive strength (MPa)",
              Water_Cement_Ratio="Water–cement ratio (kg/kg)")
SEED = 42
FOLDS = 5
TEST_SIZE = 0.2
DATA_URL = "https://archive.ics.uci.edu/static/public/165/concrete%2Bcompressive%2Bstrength.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength"
DISCLAIMER = "Academic ML demonstration only. Predictions are not certified structural assessments and do not replace laboratory testing or engineering approval."
