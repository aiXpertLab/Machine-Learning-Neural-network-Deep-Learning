import streamlit as st
from utils import st_def, tab_ml
import openai, PyPDF2, os, time, pandas as pd

st_def.st_logo(title='👋 to Machine Learning!', page_title="Machine Learning",)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Types", "Data Collection", "Preprocessing", "Feature Engineering","Python"])

with tab1:  tab_ml.ml_types()
with tab2:  tab_ml.ml_datacollection()
with tab3:  tab_ml.ml_preprocessing()
with tab4:  tab_ml.ml_featureengineering()
with tab5:  tab_ml.ml_python()



# pdf1 = st.file_uploader('Upload your PDF Document', type='pdf')
# #-----------------------------------------------
# if pdf1:
#     pdfReader = PyPDF2.PdfReader(pdf1)
#     st.session_state['pdfreader'] = pdfReader
#     st.success(" has loaded.")
# else:
#     st.info("waiting for loading ...")