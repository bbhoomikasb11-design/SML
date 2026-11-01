"""Reusable interactive EDA, deliberately independent of model selection."""
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import FEATURES, LABELS, ROOT, TARGET


def distribution(frame, column=TARGET):
    return px.histogram(frame, x=column, nbins=35, marginal="box", labels=LABELS,
                        title=f"Distribution of {LABELS[column]}", color_discrete_sequence=["#2563EB"])


def correlation(frame):
    names = ["Cement", "Slag", "Fly ash", "Water", "Plasticizer", "Coarse agg.", "Fine agg.", "Age", "Strength"]
    return px.imshow(frame[FEATURES + [TARGET]].corr(), x=names, y=names, zmin=-1, zmax=1,
                     color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto",
                     labels={"color": "Pearson r (unitless)"}, title="Pearson correlation · original variables")


def scatter(frame, feature):
    return px.scatter(frame, x=feature, y=TARGET, labels=LABELS, opacity=.55, color="Age",
                      color_continuous_scale="Teal", title=f"Observed strength versus {LABELS[feature]}",
                      hover_data={"Cement": ":.1f", "Water": ":.1f", "Age": ":.0f"})


def input_histograms(frame):
    fig = make_subplots(rows=4, cols=2, subplot_titles=[LABELS[f] for f in FEATURES], vertical_spacing=.1)
    for i, f in enumerate(FEATURES):
        fig.add_trace(go.Histogram(x=frame[f], name=LABELS[f], nbinsx=25, marker_color="#2563EB",
                                   hovertemplate="Value: %{x}<br>Count: %{y}<extra>%{fullData.name}</extra>"), row=i//2+1, col=i%2+1)
        fig.update_xaxes(title_text=LABELS[f], row=i//2+1, col=i%2+1)
        fig.update_yaxes(title_text="Observations", row=i//2+1, col=i%2+1)
    return fig.update_layout(height=1000, title="Input feature distributions", showlegend=False)


def pairwise(frame):
    return px.scatter_matrix(frame, dimensions=["Cement", "Water", "Age", TARGET], labels=LABELS,
                             opacity=.35, title="Pairwise relationships · selected variables", height=700).update_traces(diagonal_visible=False)


def export_interactive(bundle):
    """Save offline interactive EDA; the shared JS asset stays beside the HTML."""
    directory = ROOT / "artifacts/interactive"
    directory.mkdir(parents=True, exist_ok=True)
    data = bundle["data"]
    charts = {"strength_distribution": distribution(data), "input_histograms": input_histograms(data),
              "correlation": correlation(data), "pairwise": pairwise(data),
              "cement_strength": scatter(data, "Cement")}
    for name, figure in charts.items():
        figure.write_html(directory / f"{name}.html", include_plotlyjs="directory", full_html=True)
    return directory
