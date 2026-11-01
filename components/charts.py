import plotly.graph_objects as go
import streamlit as st
from textwrap import wrap


def chart(fig, key=None):
    if fig.layout.title.text:
        fig.update_layout(title_text="<br>".join(wrap(fig.layout.title.text, width=38)))
    fig.update_layout(template="plotly_white", font={"family":"DM Sans, Arial", "size":12, "color":"#334155"},
                      title_font={"size":16,"color":"#14213D"}, paper_bgcolor="white", plot_bgcolor="white",
                      margin={"l":24,"r":24,"t":70,"b":75}, colorway=["#2563EB","#14B8A6","#F59E0B"],
                      legend={"orientation":"h","y":-.22,"x":0,"title_text":""})
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo":False})


def actual_predicted(frame):
    lo, hi = min(frame.Actual.min(), frame.Predicted.min()), max(frame.Actual.max(), frame.Predicted.max())
    fig = go.Figure(go.Scatter(x=frame.Actual,y=frame.Predicted,mode="markers",name="Held-out observations",
                               marker={"color":"#2563EB","size":7,"opacity":.65},
                               hovertemplate="Actual: %{x:.2f} MPa<br>Predicted: %{y:.2f} MPa<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Perfect prediction",line={"color":"#14B8A6","dash":"dash"}))
    return fig.update_layout(title="Actual vs predicted · test set",xaxis_title="Actual strength (MPa)",yaxis_title="Predicted strength (MPa)")
