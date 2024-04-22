# from hypecheth
import streamlit as st
from utils.st_def import st_main_contents, st_logo

st_logo(title='👋 Artificial Intelligence! 🍨 ', page_title="AI🍨",)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "Learning Material", "Prediction", "Train NN","Conclusion"])

with tab1: st_main_contents()
with tab2:
    st.markdown('[FastAI Deep Learning](https://course.fast.ai/)',unsafe_allow_html=True)