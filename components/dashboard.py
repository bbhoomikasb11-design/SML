import plotly.express as px
import streamlit as st
from components.charts import actual_predicted, chart
from components.styles import page_header


def overview(bundle):
    meta = bundle["metadata"]
    row = bundle["metrics"].set_index("Model").loc[meta["selected"]]
    page_header("Statistical machine learning / overview", "Concrete strength, understood.",
                "Concrete Strength Prediction Using Polynomial Regression · SML Module 4")
    st.markdown(f'''<div class="hero"><span class="tag">SDG 9 · INDUSTRY, INNOVATION & INFRASTRUCTURE</span>
    <h2>From concrete ingredients<br>to measurable insight.</h2><p>Explore the evidence behind stronger materials. Compare linear and polynomial regression,
    examine model assumptions, and estimate strength from eight mixture properties.</p><span class="tag">SELECTED MODEL · {meta['selected'].upper()}</span></div>''',unsafe_allow_html=True)
    cols = st.columns(4)
    for col,label,value,help_text in zip(cols,["Source observations","Input features","Test RMSE","Test R²"],
        [f"{len(bundle['original']):,}","8",f"{row['Test RMSE']:.2f} MPa",f"{row['Test R2']:.3f}"],
        ["Original UCI records before duplicate removal","Seven ingredient masses and curing age","Lower is better; held-out prediction error","Fraction of test target variation explained"]):
        col.metric(label,value,help=help_text)
    st.write("")
    a,b = st.columns([1.3,1])
    with a: chart(actual_predicted(bundle["predictions"][meta["selected"]]))
    with b:
        chart(px.bar(bundle["metrics"],x="Model",y="CV RMSE",color="Model",error_y="CV SD",
                     title="Training cross-validation",labels={"CV RMSE":"Mean CV RMSE (MPa)"}).update_layout(showlegend=False))
    st.markdown(f'<div class="note">{meta["selected"]} has the lowest mean training CV RMSE ({row["CV RMSE"]:.2f} MPa). '
                'Error bars show fold standard deviation, not confidence intervals. The test set was excluded from model selection.</div>',unsafe_allow_html=True)
    st.subheader("A transparent path from data to prediction")
    for col,(n,title,desc) in zip(st.columns(4),[("01","Validate","Audit the original UCI dataset."),("02","Compare","Train three regression pipelines."),
            ("03","Diagnose","Inspect errors and collinearity."),("04","Explore","Predict, investigate and report.")]):
        col.markdown(f'<div class="step"><b>{n} / {title.upper()}</b><span>{desc}</span></div>',unsafe_allow_html=True)
