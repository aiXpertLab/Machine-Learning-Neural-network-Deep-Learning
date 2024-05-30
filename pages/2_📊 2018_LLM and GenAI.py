import streamlit as st
from utils import st_def, tab_llm

st_def.st_logo(title='👋 to Machine Learning!', page_title="Machine Learning",)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["General", "Prompt Engineering", "RAG", "Chunking","Embedding" , "Finetuning"])

with tab1:  tab_llm.llm_general()
with tab2:  tab_llm.llm_promptengineering()
with tab3:  tab_llm.llm_rag()
with tab4:  tab_llm.llm_chunking()
with tab5:  tab_llm.llm_python()
with tab6:  tab_llm.llm_finetuning()
