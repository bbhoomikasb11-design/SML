"""Dashboard pages share the exact same experiment as the academic report."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import probplot
import streamlit as st

from config import DISCLAIMER, FEATURES, LABELS, ROOT, TARGET
from components.charts import actual_predicted, chart
from components.styles import page_header
from src import eda
from src.diagnostics import calculate_diagnostics, diagnostic_interpretations
from src.feature_analysis import associations, observed_trend
from src.feature_engineering import with_analysis_features
from src.model_training import run_experiment
from src.prediction import predict_strength


def dataset(bundle):
    page_header("01 / Dataset explorer", "Start with the evidence.", "Original UCI data · mass per cubic metre · strength in MPa")
    raw, audit = bundle["original"], bundle["audit"]
    for col,label,value in zip(st.columns(4),["Original rows","Missing cells","Exact duplicates","Invalid rows"],
                               [len(raw),sum(audit["missing"].values()),audit["duplicate_rows"],audit["invalid_rows"]]):
        col.metric(label,value)
    st.info(f"The original workbook is preserved. Modeling uses {len(bundle['data'])} rows after removing exact duplicates. "
            "Repeated input vectors with different measured strengths remain grouped across train/test and CV boundaries. IQR flags are retained.")
    a,b = st.columns(2)
    age = a.slider("Age filter (days)",int(raw.Age.min()),int(raw.Age.max()),(int(raw.Age.min()),int(raw.Age.max())))
    strength = b.slider("Strength filter (MPa)",float(raw.Strength.min()),float(raw.Strength.max()),(float(raw.Strength.min()),float(raw.Strength.max())))
    filtered = raw[raw.Age.between(*age) & raw.Strength.between(*strength)]
    st.caption(f"Showing {len(filtered):,} of {len(raw):,} original observations. Filters affect this page only.")
    if filtered.empty: st.warning("No observations match these filters. Widen the ranges.")
    st.dataframe(filtered.rename(columns=LABELS),use_container_width=True,height=350)
    st.download_button("Download filtered data",filtered.to_csv(index=False).encode(),"concrete_filtered.csv","text/csv")
    summary,quality = st.tabs(["Statistical summary","Schema & data quality"])
    with summary:
        st.dataframe(filtered.describe().T.rename(index=LABELS),use_container_width=True)
    with quality:
        st.dataframe(pd.DataFrame({"Variable":[LABELS[c] for c in raw],"Type":raw.dtypes.astype(str).values,
            "Missing":[audit["missing"][c] for c in raw],"IQR outlier flags":[audit["iqr_flags"][c] for c in raw]}),hide_index=True,use_container_width=True)
        st.caption("Checks reject non-finite values, negative masses, non-positive cement/water/age/strength and non-integer ages. Missing predictor values would be median-imputed within training folds only.")
    st.markdown("[UCI dataset and attribution](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) · I-Cheng Yeh · CC BY 4.0")
    st.download_button("Download original UCI workbook",(ROOT/"data/raw/Concrete_Data.xls").read_bytes(),"Concrete_Data.xls","application/vnd.ms-excel")


def exploration(bundle):
    page_header("02 / Exploratory analysis", "Find the patterns.", "Descriptive analysis of deduplicated observations; no feature selection is based on these views.")
    data = bundle["data"]
    tabs = st.tabs(["Distributions","Correlations","Feature relationships","Outliers & pairs"])
    with tabs[0]:
        chart(eda.distribution(data))
        st.caption(f"Strength ranges from {data.Strength.min():.2f} to {data.Strength.max():.2f} MPa; median {data.Strength.median():.2f} MPa. The wide response range motivates reporting errors in physical units.")
        chart(eda.input_histograms(data))
        st.caption("Ingredient masses have different scales. Zero slag or fly ash is a valid mixture choice; age is concentrated at discrete curing periods. Scaling is learned within each training fold.")
    with tabs[1]:
        chart(eda.correlation(data))
        strongest = data[FEATURES].corrwith(data.Strength).abs().idxmax()
        st.caption(f"{strongest} has the largest absolute marginal Pearson correlation with strength in this sample. Pairwise linear association does not describe all interactions or establish causation.")
    with tabs[2]:
        feature = st.selectbox("Input variable", FEATURES,format_func=lambda f:LABELS[f])
        chart(eda.scatter(data,feature))
        st.caption("Each point is an observation; color shows curing age. Vertical spread at the same input value reflects differences in other ingredients, age and unmeasured conditions.")
    with tabs[3]:
        feature = st.selectbox("Inspect outliers", FEATURES+[TARGET],format_func=lambda f:LABELS[f])
        chart(px.box(data,y=feature,points="outliers",labels=LABELS,title=f"Box plot · {LABELS[feature]}"))
        st.caption("Points beyond 1.5 IQR whiskers are flags for inspection, not evidence of recording errors. No observations are removed solely because they are outliers.")
        chart(eda.pairwise(data))
        st.caption("The scatter matrix highlights pairwise structure. Nonlinear relationships motivate testing polynomial terms, with complexity judged through cross-validation.")


def training(bundle):
    page_header("03 / Model development", "Compare before you choose.", "Linear regression and polynomial degrees 2 and 3 · identical held-out observations")
    a,b,c = st.columns([1,1,2])
    folds = a.selectbox("Cross-validation folds",[3,5,10],index=[3,5,10].index(bundle["metadata"]["folds"]))
    inspect = b.selectbox("Inspect polynomial degree",[1,2,3],index=2,format_func=lambda v:"1 · Linear" if v==1 else str(v))
    with c:
        st.write("")
        if st.button("Train & compare all models",type="primary",use_container_width=True):
            with st.spinner("Fitting all pipelines and validating each fold…"):
                st.session_state.experiment = run_experiment(folds=folds,save=False)
                st.session_state.pop("report_paths",None)
                st.session_state.pop("last_prediction",None)
            st.rerun()
    m = bundle["metadata"]
    st.success(f"Training complete · {m['train_rows']} training rows / {m['test_rows']} test rows · {m['folds']} shuffled folds · selected {m['selected']}")
    st.caption("An 80/20 split of unique eight-input vectors prevents identical input leakage. KFold shuffles training groups; fold row counts may differ. Median imputation, scaling and polynomial expansion are fitted inside each fold. Test scores are not used to select the degree. Session retraining does not overwrite the bundled baseline.")
    st.dataframe(bundle["metrics"].round(4),hide_index=True,use_container_width=True)
    st.download_button("Download evaluation metrics",bundle["metrics"].to_csv(index=False).encode(),"model_comparison.csv","text/csv")
    a,b = st.columns(2)
    with a:
        comp = bundle["metrics"].melt(id_vars="Model",value_vars=["Train RMSE","Test RMSE","CV RMSE"],var_name="Evaluation",value_name="RMSE (MPa)")
        chart(px.bar(comp,x="Model",y="RMSE (MPa)",color="Evaluation",barmode="group",title="Prediction error comparison"))
    with b:
        chart(px.box(bundle["cv_scores"],x="Model",y="RMSE",color="Model",points="all",labels={"RMSE":"Validation RMSE (MPa)"},title="Cross-validation score distribution"))
    name = "Linear" if inspect==1 else f"Polynomial {inspect}"
    a,b = st.columns(2)
    with a: chart(actual_predicted(bundle["predictions"][name]))
    with b:
        error = pd.concat([frame.assign(Model=model) for model,frame in bundle["predictions"].items()])
        error["Absolute error (MPa)"] = error.Residual.abs()
        chart(px.box(error,x="Model",y="Absolute error (MPa)",color="Model",title="Held-out absolute errors"))
    selected = bundle["metrics"].set_index("Model").loc[m["selected"]]
    st.markdown(f"**Selection and overfitting.** {m['selected']} minimizes mean CV RMSE at {selected['CV RMSE']:.3f} MPa. "
        f"Its training RMSE is {selected['Train RMSE']:.3f} MPa versus test RMSE {selected['Test RMSE']:.3f} MPa. "
        f"The gap is {selected['Test RMSE']-selected['Train RMSE']:.3f} MPa; increasing degree reduces bias but can increase variance. "
        "Fold variability and the modest dataset size limit claims of superiority. A simpler degree 2 model may be preferable when stability or interpretation matters.")
    with st.expander("Equations, assumptions and model complexity",expanded=True):
        st.latex(r"\hat{y}=\beta_0+\sum_{j=1}^{8}\beta_j x_j")
        st.latex(r"\hat{y}=\beta_0+\sum_{1\leq |\alpha|\leq d}\beta_{\alpha}\prod_{j=1}^{8}z_j^{\alpha_j},\quad z_j=(x_j-\mu_j)/s_j")
        st.write("Polynomial regression is linear in its fitted coefficients, although nonlinear in the inputs. All powers and interactions up to degree d are included: 8, 44 and 164 terms (excluding intercept). Standardized polynomial terms are fitted by ordinary least squares.")
        st.write("Both models assume an adequate mean-function specification and independent, representative errors. Constant variance and normal errors support classical inference but are not prerequisites for least-squares point predictions. Linear regression is compact and interpretable; polynomial regression captures curvature and interactions but increases collinearity, overfitting and extrapolation risk.")


def diagnostics(bundle):
    page_header("04 / Model diagnostics", "Understand the errors.", "Held-out residuals and training-only collinearity diagnostics for the selected model")
    result = calculate_diagnostics(bundle)
    stats = result["statistics"]
    frame = bundle["predictions"][bundle["metadata"]["selected"]]
    residual_tab,vif_tab = st.tabs(["Residual diagnostics","Multicollinearity"])
    with residual_tab:
        a,b = st.columns(2)
        with a:
            fig = px.scatter(frame,x="Predicted",y="Residual",trendline="lowess",labels={"Predicted":"Predicted strength (MPa)","Residual":"Actual − predicted (MPa)"},title="Residuals versus predicted strength")
            chart(fig.add_hline(y=0,line_dash="dash"))
            chart(px.histogram(frame,x="Residual",nbins=30,labels={"Residual":"Residual (MPa)"},title="Held-out residual distribution"))
        with b:
            (theoretical,ordered),(slope,intercept,r) = probplot(frame.Residual,dist="norm")
            fig = go.Figure(go.Scatter(x=theoretical,y=ordered,mode="markers",name="Residual quantiles"))
            fig.add_trace(go.Scatter(x=theoretical,y=slope*theoretical+intercept,mode="lines",name="Normal reference"))
            chart(fig.update_layout(title="Normal Q–Q plot",xaxis_title="Theoretical normal quantile (unitless)",yaxis_title="Ordered residual (MPa)"))
            chart(px.scatter(frame,x="Actual",y="Residual",labels={"Actual":"Actual strength (MPa)","Residual":"Residual (MPa)"},title="Residuals versus actual strength").add_hline(y=0,line_dash="dash"))
        for text in diagnostic_interpretations(stats): st.info(text)
        st.caption("A curved LOWESS residual trend can indicate misspecification; a funnel can indicate non-constant variance. Residuals versus actual values have inherent mathematical coupling, so correlation there is not an independent specification test.")
        st.dataframe(pd.DataFrame(list(stats.items()),columns=["Diagnostic","Value"]),hide_index=True,use_container_width=True)
        st.subheader("Largest held-out errors")
        st.dataframe(result["unusual"].round(3),use_container_width=True)
        st.caption("Source row indices are retained. These are large prediction errors, not formal influence or leverage measures; they are not removed.")
    with vif_tab:
        a,b = st.columns([1,1.6])
        with a:
            st.dataframe(result["vif"].round(3),hide_index=True,use_container_width=True)
            chart(px.bar(result["vif"],x="VIF",y="Feature",orientation="h",title="Variance inflation factors").add_vline(x=5,line_dash="dash"))
        with b: chart(eda.correlation(bundle["data"].iloc[bundle["train_indices"]]))
        st.write("VIF uses the eight original predictors from the training partition plus an intercept. For each predictor, VIF = 1 / (1 − R²) from regressing it on the other predictors. The intercept's VIF is omitted. Values above 5 are caution flags, not automatic removal rules.")
        high = result["vif"].query("VIF > 5").Feature.tolist()
        st.write("Variables with VIF > 5: " + (", ".join(high) if high else "none") + ".")
        if result["high_correlations"].empty: st.caption("No training predictor pair has |Pearson r| ≥ 0.70. Joint dependencies can still increase VIF.")
        else: st.dataframe(result["high_correlations"],hide_index=True)
        st.write(f"The expanded selected design has rank {stats['Expanded design rank']} of {stats['Expanded design columns']} columns and condition number {stats['Expanded condition number']:.2e}. Original-variable VIF does not measure polynomial-term collinearity. Scaling improves numerical conditioning but does not remove dependencies. All scientifically relevant predictors are retained; coefficients should not be interpreted as isolated causal effects. Ridge regression is a future comparison, not a replacement for the required OLS experiment.")


def feature_analysis(bundle):
    page_header("05 / Feature analysis", "Water, time and strength.", "Observed associations and model scenarios are shown separately")
    data = with_analysis_features(bundle["data"])
    stats = associations(data)
    observed,scenario = st.tabs(["Observed relationships","Interactive model scenarios"])
    with observed:
        st.dataframe(stats.round(5),hide_index=True,use_container_width=True)
        a,b = st.columns(2)
        with a:
            chart(observed_trend(data,"Water_Cement_Ratio"))
            chart(eda.distribution(data,"Water_Cement_Ratio"))
        with b:
            chart(observed_trend(data,"Age"))
            chart(px.box(data,x="Age Group",y=TARGET,color="Age Group",labels=LABELS,title="Observed strength across curing periods"))
        st.dataframe(data.groupby("Age Group",observed=True).Strength.agg(["count","mean","median","std"]).rename(columns={"mean":"Mean (MPa)","median":"Median (MPa)","std":"SD (MPa)"}),use_container_width=True)
        st.write("Water–cement ratio is water mass divided by cement mass; it excludes slag and fly ash from the denominator. Excess water relative to cement can leave a more porous hardened paste. Hydration generally develops strength over time when moisture and temperature are suitable. The dataset records age, not curing conditions; these observational comparisons are confounded by composition and do not establish causation. LOWESS lines describe sample patterns, not the trained prediction model.")
        st.markdown("Background: [American Cement Association](https://www.cement.org/cement-concrete/applications-of-cement/) · [ACI curing guidance](https://www.concrete.org/portals/0/files/pdf/previews/308r_16_preview.pdf)")
    with scenario:
        train = bundle["data"].iloc[bundle["train_indices"]]
        st.caption("Start from a real training mixture. Vary water at fixed cement to change the ratio; vary age while keeping all other mixture masses fixed. Combinations may still be uncommon even inside marginal ranges.")
        source_row = st.selectbox("Reference mixture (source row)",train.index.tolist(),index=0)
        base = train.loc[source_row,FEATURES].to_dict()
        st.dataframe(pd.DataFrame([base]).rename(columns=LABELS),hide_index=True,use_container_width=True)
        a,b = st.columns(2)
        ratio_min,ratio_max = train.Water.min()/base["Cement"],train.Water.max()/base["Cement"]
        ratio = a.slider("Water–cement ratio (kg/kg)",float(ratio_min),float(ratio_max),float(base["Water"]/base["Cement"]),step=.001)
        age = b.slider("Scenario age (days)",int(train.Age.min()),int(train.Age.max()),int(base["Age"]))
        base.update(Water=ratio*base["Cement"],Age=float(age))
        result = predict_strength(bundle,base)
        st.metric("Model scenario prediction",f"{result['Prediction (MPa)']:.2f} MPa")
        for warning in result["warnings"]: st.warning(warning)
        st.caption(f"Derived water = {base['Water']:.2f} kg/m³; cement stays at {base['Cement']:.2f} kg/m³. Curves below are model predictions, not measured curing histories or causal effects.")
        model = bundle["models"][bundle["metadata"]["selected"]]
        ratio_grid = np.linspace(ratio_min,ratio_max,60)
        age_grid = np.unique(np.linspace(train.Age.min(),train.Age.max(),90).astype(int))
        a,b = st.columns(2)
        for col,feature,grid,xlabel in [(a,"Water",ratio_grid,"Water–cement ratio (kg/kg)"),(b,"Age",age_grid,"Age (days)")]:
            rows = pd.DataFrame([base]*len(grid),columns=FEATURES)
            rows[feature] = grid*base["Cement"] if feature=="Water" else grid
            curve = pd.DataFrame({xlabel:grid,"Predicted strength (MPa)":model.predict(rows)})
            with col: chart(px.line(curve,x=xlabel,y="Predicted strength (MPa)",title="Model response · other inputs fixed"))
        st.warning("Polynomial response curves are unconstrained: they can be non-monotonic or physically implausible, especially at sparse ages. This tool cannot optimize or certify a concrete mix.")


def prediction(bundle):
    page_header("06 / Strength prediction", "Explore your concrete mix.", "Enter ingredient masses per cubic metre and curing age to estimate compressive strength")
    train = bundle["data"].iloc[bundle["train_indices"]]
    candidates = {"28-day mixture":int((train.Age-28).abs().idxmin()),"7-day mixture":int((train.Age-7).abs().idxmin()),
                  "90-day mixture":int((train.Age-90).abs().idxmin())}
    default = train.loc[candidates["28-day mixture"],FEATURES].to_dict()
    def populate(values):
        for f in FEATURES: st.session_state[f"input_{f}"] = float(values[f])
        st.session_state.pop("last_prediction",None)
    if "input_Cement" not in st.session_state: populate(default)
    a,b,c = st.columns([2,1,1])
    a.selectbox("Sample from training data",list(candidates),key="preset_choice")
    def load_selected_preset():
        populate(train.loc[candidates[st.session_state.preset_choice],FEATURES].to_dict())
    b.button("Load preset",on_click=load_selected_preset,use_container_width=True)
    c.button("Reset inputs",on_click=populate,args=(default,),use_container_width=True)
    left,right = st.columns([1.65,1])
    with left:
        with st.form("predict_form",border=True):
            st.subheader("Mixture composition")
            cols = st.columns(2)
            values = {}
            for i,f in enumerate(FEATURES):
                with cols[i%2]:
                    values[f] = st.number_input(LABELS[f],min_value=0.0,step=1.0 if f=="Age" else .1,
                        key=f"input_{f}",format="%.1f",help=f"Training range: {train[f].min():.1f}–{train[f].max():.1f}. Outside values trigger extrapolation warnings.")
            submitted = st.form_submit_button("Predict compressive strength",type="primary",use_container_width=True)
        if submitted:
            try:
                result = predict_strength(bundle,values)
                st.session_state.last_prediction = result
                record = {"Time (UTC)":datetime.now(timezone.utc).isoformat(),**values,**{k:v for k,v in result.items() if k!="warnings"},"Warnings":"; ".join(result["warnings"])}
                st.session_state.setdefault("history",[]).append(record)
            except ValueError as exc:
                st.session_state.pop("last_prediction",None)
                st.error(str(exc))
    with right:
        with st.container(border=True):
            st.subheader("Prediction result")
            result = st.session_state.get("last_prediction")
            if result:
                st.metric("Estimated compressive strength",f"{result['Prediction (MPa)']:.2f} MPa")
                st.write(f"**Model:** {result['Model']}")
                st.write(f"**Water–cement ratio:** {result['Water/Cement']:.3f} kg/kg")
                st.caption("Result for the last submitted mixture. Test RMSE describes aggregate model error; it is not a confidence interval for this individual prediction.")
                for warning in result["warnings"]: st.warning(warning)
            else: st.info("Enter a mix and select Predict to see the estimated strength.")
            st.caption(DISCLAIMER)
    st.subheader("Session prediction history")
    history = pd.DataFrame(st.session_state.get("history",[]))
    if history.empty: st.caption("Your predictions will appear here. History stays in this browser session only.")
    else:
        st.dataframe(history,use_container_width=True,hide_index=True)
        st.download_button("Download predictions as CSV",history.to_csv(index=False).encode(),"concrete_predictions.csv","text/csv")
        if st.button("Clear session history"):
            st.session_state.history=[]
            st.rerun()


def report(bundle):
    page_header("07 / Academic report", "From analysis to submission.", "A complete report generated from the active experiment, with editable Word and print-ready PDF outputs")
    st.markdown("**Concrete Strength Prediction Using Polynomial Regression**  \nBhoomika S · Artificial Intelligence and Data Science  \nDon Bosco Institute of Technology, Bengaluru · SML Module 4 · SDG 9")
    st.info(f"Active experiment: {bundle['metadata']['experiment_id']} · {bundle['metadata']['selected']} · {bundle['metadata']['folds']}-fold CV. Both reports use this run's actual metrics and figures.")
    with st.expander("Report contents",expanded=True):
        st.write("Title page and contents; abstract; introduction; problem statement; objectives; literature review; dataset; requirements; methodology; preprocessing; EDA; linear and polynomial implementations; cross-validation; residual diagnostics; multicollinearity; water–cement ratio; curing age; results; conclusion; limitations; future scope; references.")
    if st.button("Generate PDF & editable Word report",type="primary"):
        from reports.generate_report import generate_reports
        with st.spinner("Generating figures and academic reports from the active experiment…"):
            st.session_state.report_paths = generate_reports(bundle)
        st.success("Both reports generated successfully.")
    paths = st.session_state.get("report_paths")
    if not paths:
        out = ROOT/"reports/output"/bundle["metadata"]["experiment_id"]
        if (out/"concrete_strength_report.pdf").exists() and (out/"concrete_strength_report.docx").exists():
            paths = {"pdf":out/"concrete_strength_report.pdf","docx":out/"concrete_strength_report.docx"}
    if paths:
        a,b = st.columns(2)
        for col,kind,mime in [(a,"pdf","application/pdf"),(b,"docx","application/vnd.openxmlformats-officedocument.wordprocessingml.document")]:
            with col: st.download_button(f"Download {kind.upper()} report",paths[kind].read_bytes(),paths[kind].name,mime,use_container_width=True)
    st.caption("Certificate, declaration and acknowledgement pages are omitted because no official institutional templates were provided.")
