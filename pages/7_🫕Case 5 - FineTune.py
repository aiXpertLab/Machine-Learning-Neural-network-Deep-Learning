import streamlit as st
from streamlit_extras.stateful_button import button
from utils import st_def, tab_dogs_cats

st_def.st_logo(title = "👋Dogs and Cats", page_title="Summary",)

t1, t2, t3, t4, t5, t6 = st.tabs(["Llama3", " ", " ", '', '',''])

with t1: 
    # https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations
    st.subhead('Step 1—Installation')
    st.write("""
        - transformers: to load models, tokenizers, etc.
        - peft: to perform parameter efficient fine-tuning
        - bitsandbytes: to setup 4-bit quantization
        - trl: for supervised fine-tuning
        - deepeval: to evaluate the fine-tuned LLM
    
    
    
    """)
    