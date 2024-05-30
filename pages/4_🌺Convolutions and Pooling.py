import streamlit as st, time
from utils import st_def, tab_convolutions_pooling

st_def.st_logo(title = "👋 Convolutions and Pooling!", page_title="Summary",)
t1, t2, t3, t4, t5, t6 = st.tabs(["General", "Convolution in CNN, Vision, Leanred Filters", "Convolutional Layers", "","" , "🧹"])

with t1: tab_convolutions_pooling.cp_general()
with t2: tab_convolutions_pooling.cp_convolution_cnn()
with t3: tab_convolutions_pooling.cp_convolutional_layers()