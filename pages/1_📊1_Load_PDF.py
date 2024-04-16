import streamlit as st
from utils.st_def import st_logo, st_load_book
import openai, PyPDF2, os, time, pandas as pd

st_logo(title='Welcome 👋 to Book Summarizer!', page_title="PDF Summarizer",)
st_load_book()

pdf1 = st.file_uploader('Upload your PDF Document', type='pdf')
#-----------------------------------------------
if pdf1:
    pdfReader = PyPDF2.PdfReader(pdf1)
    st.session_state['pdfreader'] = pdfReader
    st.success(" has loaded.")
else:
    st.info("waiting for loading ...")