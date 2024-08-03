import streamlit as st
from prediction_page import show_predict_page
from diameter_page import show_diameter_page

st.set_page_config(layout="wide")

page = st.sidebar.selectbox("Choose a page", ["Defect Detection", "Diameter Measurement"])

if page == "Defect Detection":
    show_predict_page()
else:
    show_diameter_page()
