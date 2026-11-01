import pandas as pd
from scipy.stats import pearsonr, spearmanr
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import LABELS, TARGET
from src.feature_engineering import with_analysis_features


def associations(frame):
    enriched = with_analysis_features(frame)
    rows = []
    for feature in ["Water_Cement_Ratio", "Age"]:
        clean = enriched[[feature, TARGET]].dropna()
        pearson, pearson_p = pearsonr(clean[feature], clean[TARGET])
        spearman, spearman_p = spearmanr(clean[feature], clean[TARGET])
        rows.append({"Feature": feature, "Pearson r": pearson, "Pearson p": pearson_p,
                     "Spearman rho": spearman, "Spearman p": spearman_p, "n": len(clean)})
    return pd.DataFrame(rows)


def observed_trend(frame, feature):
    fig = px.scatter(frame, x=feature, y=TARGET, labels=LABELS, opacity=.35,
                     title=f"Observed {LABELS[feature]} and strength", color_discrete_sequence=["#2563EB"])
    # A wider age neighborhood spans more than the large tied 28-day block.
    curve = lowess(frame[TARGET], frame[feature], frac=.6 if feature == "Age" else .35, return_sorted=True)
    fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines", name="LOWESS descriptive trend",
                             line={"color": "#0d9488", "width": 3}))
    return fig
