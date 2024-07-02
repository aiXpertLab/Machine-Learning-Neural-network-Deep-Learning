# from hypecheth
import streamlit as st
from utils import st_def, tab_home

st_def.st_logo(title='👋 Artificial Intelligence! 🍨 ', page_title="AI🍨",)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "Learning ", "Developing Environment", "Developing Process",""])

with tab1: tab_home.home_main_contents()
with tab2: st.markdown('[FastAI Deep Learning](https://course.fast.ai/)',unsafe_allow_html=True)
with tab3: tab_home.home_dev()
with tab4: tab_home.home_process()
