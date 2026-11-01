"""One reproducible experiment, consumed by both dashboard and reports."""
import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
from threadpoolctl import threadpool_limits

from config import FEATURES, FOLDS, ROOT, SEED, TARGET
from src.data_loader import WORKBOOK, load_data, prepare_data, save_dataset_audit
from src.evaluation import regression_metrics
from src.preprocessing import make_pipeline, make_splits

BUNDLE_PATH = ROOT / "models/experiment.joblib"


def run_experiment(folds=FOLDS, seed=SEED, save=True):
    original = load_data()
    provenance = save_dataset_audit(original)
    data = prepare_data(original)
    train, test, cv, groups = make_splits(data, folds, seed)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
    models, scores, predictions, fold_scores = {}, [], {}, []
    with threadpool_limits(limits=1):
        for degree, name in [(1, "Linear"), (2, "Polynomial 2"), (3, "Polynomial 3")]:
            pipeline = make_pipeline(degree)
            cv_result = cross_validate(pipeline, X_train, y_train, cv=cv,
                                       scoring="neg_root_mean_squared_error", return_train_score=True,
                                       n_jobs=1, error_score="raise")
            pipeline.fit(X_train, y_train)
            train_pred, test_pred = pipeline.predict(X_train), pipeline.predict(X_test)
            row = {"Model": name, "Degree": degree,
                   "Terms": pipeline.named_steps["polynomial"].n_output_features_,
                   "CV RMSE": float(-cv_result["test_score"].mean()),
                   "CV SD": float(cv_result["test_score"].std(ddof=1))}
            for partition, actual, pred in [("Train", y_train, train_pred), ("Test", y_test, test_pred)]:
                row.update({f"{partition} {metric}": value for metric, value in regression_metrics(actual, pred).items()})
            scores.append(row)
            for i, value in enumerate(-cv_result["test_score"]):
                fold_scores.append({"Model": name, "Fold": i+1, "RMSE": float(value),
                                    "Training RMSE": float(-cv_result["train_score"][i]),
                                    "Validation rows": len(cv[i][1])})
            models[name] = pipeline
            predictions[name] = pd.DataFrame({"Actual": y_test, "Predicted": test_pred,
                                               "Residual": y_test - test_pred}, index=y_test.index)
    metrics = pd.DataFrame(scores)
    # Degrees were prespecified. Tie-break with fewer terms. Holdout never used here.
    selected = metrics.sort_values(["CV RMSE", "Terms"]).iloc[0]["Model"]
    metadata = {"seed": seed, "folds": folds, "train_rows": len(train), "test_rows": len(test),
                "modeling_rows": len(data), "unique_input_groups": len(np.unique(groups)),
                "selected": selected, "selection": "Lowest mean training CV RMSE; fewer terms breaks ties",
                "split": "80/20 unique input groups; shuffled KFold on training groups",
                "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "sklearn": sklearn.__version__, "data_sha256": provenance["sha256"]}
    metadata["experiment_id"] = hashlib.sha256(json.dumps({k: metadata[k] for k in
        ["seed", "folds", "data_sha256", "sklearn"]}, sort_keys=True).encode()).hexdigest()[:12]
    bundle = {"metadata": metadata, "original": original, "data": data, "train_indices": train,
              "test_indices": test, "cv_splits": cv, "groups": groups, "models": models,
              "metrics": metrics, "predictions": predictions, "cv_scores": pd.DataFrame(fold_scores),
              "audit": provenance["audit"]}
    if save:
        save_experiment(bundle)
    return bundle


def save_experiment(bundle):
    for directory in ["models", "artifacts/metrics", "artifacts/model_metadata"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE_PATH.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(BUNDLE_PATH)
    joblib.dump(bundle["models"][bundle["metadata"]["selected"]], ROOT / "models/selected_pipeline.joblib")
    bundle["metrics"].to_csv(ROOT / "artifacts/metrics/model_comparison.csv", index=False)
    bundle["cv_scores"].to_csv(ROOT / "artifacts/metrics/cv_folds.csv", index=False)
    for name, frame in bundle["predictions"].items():
        frame.to_csv(ROOT / f"artifacts/metrics/{name.lower().replace(' ', '_')}_predictions.csv", index_label="source_row")
    (ROOT / "artifacts/model_metadata/experiment.json").write_text(json.dumps(bundle["metadata"], indent=2))
    assignment = bundle["data"].copy()
    assignment["partition"] = "test"
    assignment.iloc[bundle["train_indices"], assignment.columns.get_loc("partition")] = "train"
    assignment.to_csv(ROOT / "data/processed/split_assignment.csv", index_label="source_row")


def load_experiment():
    if not BUNDLE_PATH.exists():
        return run_experiment()
    # Only load this application's locally generated, trusted joblib artifacts.
    bundle = joblib.load(BUNDLE_PATH)
    if bundle["metadata"]["sklearn"] != sklearn.__version__:
        raise ValueError("Model version mismatch. Run python -m src.model_training to rebuild.")
    if WORKBOOK.exists() and hashlib.sha256(WORKBOOK.read_bytes()).hexdigest() != bundle["metadata"]["data_sha256"]:
        raise ValueError("Source data changed. Retrain before using the saved experiment.")
    return bundle


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, choices=[3, 5, 10], default=5)
    args = parser.parse_args()
    result = run_experiment(folds=args.folds)
    print(result["metrics"].round(4).to_string(index=False))
    print("Selected:", result["metadata"]["selected"])
