import streamlit as st
from utils import st_def, st_DL

st_def.st_logo(title = "👋Transformer Stage 1", page_title="Data Collections",)
st_DL.st_dl1()
st.header("Data Acquisition from yfinance")
# with st.spinner(text="Checking Tensorflow and loading Apple Data  ..."):
#------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply

def main():
    with st.spinner(text="Checking Tensorflow and loading Apple Data  ..."):
        aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')        # Fetch AAPL data
        aapl_data.isnull().sum()        # Checking for missing values
        aapl_data.fillna(method='ffill', inplace=True)        # Filling missing values, if any
        if 'aapl_data' not in st.session_state:   
            st.session_state['aapl_data'] = ''
        st.session_state['aapl_data'] = aapl_data
    st.text(aapl_data.head())
    st.success("TensorFlow Version: "+ tf.__version__)

if __name__ == "__main__":
    main()
