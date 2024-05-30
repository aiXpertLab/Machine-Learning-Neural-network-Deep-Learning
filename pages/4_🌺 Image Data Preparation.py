import streamlit as st, time
from utils import st_def, tab_image_data_preparation

st_def.st_logo(title = "👋 Image Data Preparation!", page_title="Summary",)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Load Image", "Scale Image Pixel Data", "Load Large Dataset", "Data Augmentation","Kaldi" , "🧹"])

with tab1: tab_image_data_preparation.idp_load()
with tab2: tab_image_data_preparation.idp_scale()
with tab3: tab_image_data_preparation.idp_lld()