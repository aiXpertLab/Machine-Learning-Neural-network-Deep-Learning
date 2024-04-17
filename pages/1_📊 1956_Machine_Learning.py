import streamlit as st
from utils.st_def import st_logo, st_load_ML
import openai, PyPDF2, os, time, pandas as pd

st_logo(title='👋 to Machine Learning!', page_title="Machine Learning",)
st_load_ML()

# pdf1 = st.file_uploader('Upload your PDF Document', type='pdf')
# #-----------------------------------------------
# if pdf1:
#     pdfReader = PyPDF2.PdfReader(pdf1)
#     st.session_state['pdfreader'] = pdfReader
#     st.success(" has loaded.")
# else:
#     st.info("waiting for loading ...")