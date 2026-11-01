"""Run with: streamlit run app.py"""
import streamlit as st
from components.styles import apply_styles
from components.sidebar import navigation
from components.dashboard import overview
from components import pages
from config import DISCLAIMER
from src.model_training import load_experiment

st.set_page_config(page_title="ConcreteIQ · Strength Prediction",page_icon="◈",layout="wide",initial_sidebar_state="expanded")
apply_styles()


@st.cache_resource
def initial_experiment():
    return load_experiment()


try:
    if "experiment" not in st.session_state:
        with st.spinner("Loading the validated experiment…"):
            st.session_state.experiment = initial_experiment()
    bundle = st.session_state.experiment
    selected_page = navigation(bundle)
    routes = {"Overview Dashboard":overview,"Dataset Explorer":pages.dataset,"Exploratory Data Analysis":pages.exploration,
              "Model Training":pages.training,"Model Diagnostics":pages.diagnostics,"Feature Analysis":pages.feature_analysis,
              "Strength Prediction":pages.prediction,"Project Report":pages.report}
    routes[selected_page](bundle)
    st.divider()
    st.caption(DISCLAIMER)
except (ValueError,RuntimeError,OSError) as exc:
    st.error(f"Unable to complete this operation: {exc}")
    st.info("Check the local dataset and dependencies. Rebuild with `python -m src.model_training`, then restart the application.")
