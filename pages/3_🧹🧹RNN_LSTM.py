import streamlit as st
from utils import st_def, st_DL

st_def.st_logo(title = "👋Prediction using LSTM with the Attention Mechanism in TensorFlow", page_title="Text Cleaning",)
st_DL.st_dl1()
#------------------------------------------------------------------------
# import openai, PyPDF2, os, time, pandas as pd

# if 'pdfreader' not in st.session_state:   
#     st.error('Load PDF before continue ... ')
# else:
#     page_text=[]     #array for page
#     summary=' '
#     pr = st.session_state['pdfreader']
#     with st.spinner('Loading files...'):
#         for i in range(0,len(pr.pages)):
#             # creating a page object
#             pageObj = pr.pages[i].extract_text()    # extract one page's text
#             pageObj= pageObj.replace('\t\r','')     # tab, enter
#             pageObj= pageObj.replace('\xa0','')     # non-breaking spaces
#             # extracting text from page
#             page_text.append(pageObj)                    # the whole pdf --> txt
            
#     st.session_state['page_text'] = page_text
#     st.write(page_text)