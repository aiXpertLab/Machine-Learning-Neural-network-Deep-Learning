import streamlit as st, time
from utils import st_def, tab_transformer

st_def.st_logo(title = "👋 Convolutions and Pooling!", page_title="Summary",)
t1, t2, t3, t4, t5, t6 = st.tabs(["General Attention", "General Atttion in Vector", "", "","" , "🧹"])

with t1: tab_transformer.tf_attention()
with t2: tab_transformer.tf_attentioninvector()

