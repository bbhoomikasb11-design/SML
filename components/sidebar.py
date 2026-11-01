import streamlit as st

PAGES = ["Overview Dashboard", "Dataset Explorer", "Exploratory Data Analysis", "Model Training",
         "Model Diagnostics", "Feature Analysis", "Strength Prediction", "Project Report"]


def navigation(bundle):
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-icon">◈</span> ConcreteIQ</div>', unsafe_allow_html=True)
        st.caption("MATERIAL INTELLIGENCE LAB")
        st.divider()
        page = st.radio("WORKSPACE", PAGES, label_visibility="visible")
        st.divider()
        st.caption("ACTIVE EXPERIMENT")
        st.markdown(f"**{bundle['metadata']['selected']}**")
        st.caption(f"{bundle['metadata']['folds']}-fold validation · seed {bundle['metadata']['seed']}")
        st.caption(f"Run {bundle['metadata']['experiment_id']}")
        st.divider()
        st.markdown("**Bhoomika S**")
        st.caption("AI & Data Science\n\nDon Bosco Institute of Technology\n\nSML · Module 4 · SDG 9")
    return page
