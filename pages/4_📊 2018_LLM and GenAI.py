import streamlit as st
from utils import st_def, tab_llm
import openai, PyPDF2, os, time, pandas as pd

st_def.st_logo(title='👋 to Machine Learning!', page_title="Machine Learning",)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["General", "Prompt Engineering", "RAG", "Chunking","Embedding" , "VectorDB"])

with tab1:  tab_llm.llm_general()
with tab2:  tab_llm.llm_promptengineering()
with tab3:  tab_llm.llm_rag()
with tab4:  tab_llm.llm_chunking()
with tab5:  tab_llm.llm_python()
with tab6: pass
    


# pdf1 = st.file_uploader('Upload your PDF Document', type='pdf')
# #-----------------------------------------------
# if pdf1:
#     pdfReader = PyPDF2.PdfReader(pdf1)
#     st.session_state['pdfreader'] = pdfReader
#     st.success(" has loaded.")
# else:
#     st.info("waiting for loading ...")