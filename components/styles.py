import streamlit as st


def apply_styles():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
    h1,h2,h3 {font-family:'Manrope',sans-serif !important; letter-spacing:-.035em;}
    h1 {font-size:2.65rem !important; font-weight:800 !important; color:#14213D;}
    h2 {font-size:1.45rem !important;} h3 {font-size:1.15rem !important;}
    .block-container {padding-top:3.6rem; padding-bottom:3rem; max-width:1480px;}
    [data-testid="stSidebar"] {background:#14213D; border-right:0;}
    [data-testid="stSidebar"] * {color:#E2E8F0;}
    [data-testid="stSidebar"] .stCaptionContainer p {color:#A6B7D4;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:7px 8px; border-radius:9px;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.7rem;}
    [data-testid="stSidebar"] hr {margin:1rem 0;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#263B60;}
    [data-testid="stMetric"] {background:white;border:1px solid #E2E8F0;border-radius:14px;padding:18px 21px;box-shadow:0 3px 12px #14213d04;}
    [data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-size:1.9rem;font-weight:800;color:#14213D;}
    [data-testid="stMetricLabel"] {color:#64748B;}
    [data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:14px;}
    .eyebrow {font-size:.73rem;letter-spacing:.15em;color:#64748B;font-weight:700;text-transform:uppercase;margin-bottom:8px;}
    .hero {border-radius:20px;background:linear-gradient(115deg,#14213D,#203E6A);padding:32px 36px;color:#F8FAFC;margin:12px 0 25px;position:relative;overflow:hidden;}
    .hero h2 {color:white !important;font-size:2rem !important;max-width:650px;margin:8px 0;}
    .hero p {color:#CBD5E1;max-width:670px;font-size:.96rem;line-height:1.7;}
    .tag {display:inline-block;font-size:.7rem;border:1px solid #51728B;border-radius:20px;padding:5px 11px;color:#96F0DB;letter-spacing:.07em;}
    .brand {font-family:Manrope,sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:-.04em;margin:14px 0 0;}
    .brand-icon {color:#5EEAD4 !important;margin-right:7px;}
    .note {border-left:3px solid #14B8A6;padding:10px 16px;background:#ECFDF5;border-radius:0 8px 8px 0;color:#155E55;font-size:.9rem;margin:8px 0 20px;}
    .step {padding:15px;background:white;border:1px solid #E2E8F0;border-radius:12px;min-height:115px;}
    .step b {display:block;color:#2563EB;font-size:.75rem;margin-bottom:8px;}.step span {font-size:.84rem;color:#64748B;}
    .stButton>button {border-radius:9px;font-weight:600;}
    [data-testid="stPlotlyChart"] {background:white;border-radius:12px;}
    footer {visibility:hidden;}
    @media(max-width:700px){h1{font-size:1.9rem !important;}.hero{padding:22px;}.hero h2{font-size:1.5rem !important;}.block-container{padding:3.9rem 1rem 1.4rem;}}
    </style>""", unsafe_allow_html=True)


def page_header(label, title, description):
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)
