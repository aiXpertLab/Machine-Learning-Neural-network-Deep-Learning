import streamlit as st
from utils import st_def, tab_dl

st_def.st_logo(title = "👋Integrating the Attention Mechanism", page_title="Attentions",)
tab_dl.st_dl4()
#------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply

def main():
    # st.session_state
    if 'X_train' not in st.session_state:
        st.info("Click: '2. Transformer - Cleaning and Feature Engineering' first.")
    else:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']

        if 'model' not in st.session_state:
            st.info("Click: '3. Creating LSTM Layers' first.")
        else:
            model = st.session_state['model']

            from keras.layers import BatchNormalization

            # Adding Dropout and Batch Normalization
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.summary()

            st.success('This custom layer computes a weighted sum of the input sequence, allowing the model to pay more attention to certain time steps.')
            st.success("TensorFlow Version: "+ tf.__version__)        

if __name__ == "__main__":
    main()