import streamlit as st
from utils import st_def, tab_dl
st_def.st_logo(title = "2006 👋 Deep Learning!", page_title="2006 Deep Learning",)

tab1, tab2, tab3, t4, t5, t6 = st.tabs(["General", "Theory", "Vision", 'CNN', 'MLP',''])

with tab1:  tab_dl.dl_general()
with tab2:  tab_dl.dl_theory()
with tab3:  tab_dl.dl_vision()
with t4: tab_dl.dl_cnn()
with t5: tab_dl.dl_mlp()