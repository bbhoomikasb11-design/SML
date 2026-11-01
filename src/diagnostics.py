"""Descriptive holdout diagnostics and training-design collinearity checks."""
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import FEATURES


def calculate_diagnostics(bundle):
    selected = bundle["metadata"]["selected"]
    pred = bundle["predictions"][selected]
    X_train = bundle["data"].iloc[bundle["train_indices"]][FEATURES]
    X_test = bundle["data"].iloc[bundle["test_indices"]][FEATURES]
    # Intercept is essential; fit medians on training observations only.
    filled = X_train.fillna(X_train.median())
    design = sm.add_constant(filled, has_constant="add")
    vif = pd.DataFrame({"Feature": FEATURES,
                        "VIF": [variance_inflation_factor(design.to_numpy(), i+1) for i in range(len(FEATURES))]})
    residual = pred.Residual.to_numpy()
    bp_design = sm.add_constant(X_test.fillna(X_train.median()), has_constant="add")
    lm, lm_p, fstat, f_p = het_breuschpagan(residual, bp_design, robust=True)
    sw, sw_p = shapiro(residual)
    pipe = bundle["models"][selected]
    transformed = pipe[:-1].transform(X_train)
    full_design = sm.add_constant(transformed, has_constant="add")
    stats = {"BP LM": float(lm), "BP p": float(lm_p), "BP F": float(fstat), "BP F p": float(f_p),
             "Shapiro W": float(sw), "Shapiro p": float(sw_p), "Residual mean": float(residual.mean()),
             "Residual SD": float(residual.std(ddof=1)),
             "Beyond 3 SD": int((np.abs(residual-residual.mean()) > 3*residual.std(ddof=1)).sum()),
             "Expanded design rank": int(np.linalg.matrix_rank(full_design)),
             "Expanded design columns": full_design.shape[1],
             "Expanded condition number": float(np.linalg.cond(full_design))}
    pairs = []
    corr = filled.corr()
    for i, feature in enumerate(FEATURES):
        for other in FEATURES[i+1:]:
            if abs(corr.loc[feature, other]) >= .7:
                pairs.append({"Feature 1": feature, "Feature 2": other, "Pearson r": corr.loc[feature, other]})
    return {"statistics": stats, "vif": vif, "high_correlations": pd.DataFrame(pairs),
            "unusual": pred.assign(Absolute_error=pred.Residual.abs()).nlargest(10, "Absolute_error")}


def diagnostic_interpretations(stats):
    bp = ("Evidence of non-constant error variance" if stats["BP p"] < .05
          else "No statistically significant evidence of non-constant variance at the 5% level; this does not establish constant variance")
    normal = ("The Shapiro–Wilk test rejects normality at the 5% level" if stats["Shapiro p"] < .05
              else "The Shapiro–Wilk test does not reject normality at the 5% level; this is not proof of normality")
    return (f"{bp} (Koenker / robust Breusch–Pagan p = {stats['BP p']:.4g}). "
            "This exploratory holdout diagnostic regresses squared prediction errors on an intercept and the eight original predictors. "
            "It is not a classical in-sample OLS specification test, and repeated formulations may weaken independence.",
            f"{normal} (p = {stats['Shapiro p']:.4g}). Inspect the Q–Q plot for tail departures. "
            "Normal errors are not required for point prediction; distributional assumptions matter for classical inference.")
