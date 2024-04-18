import streamlit as st
from utils import st_def, st_DL

st_def.st_logo(title = "👋Transformer 2: Feature Engineering", page_title="Text Cleaning",)
st_DL.st_dl2()
#------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def main():
    # st.session_state
    if 'aapl_data' not in st.session_state:
        st.info("Click: '1. Transformer - Data Collection' first.")
    else:
        aapl_data=st.session_state['aapl_data']

        scaler = MinMaxScaler(feature_range=(0,1))      #Applying Min-Max Scaling: This scales the dataset so that all the input features lie between 0 and 1.
        aapl_data_scaled = scaler.fit_transform(aapl_data['Close'].values.reshape(-1,1))

        st.success(' Creating Sequences')
        X = []  #LSTM models require input to be in a sequence format. We transform the data into sequences for the model to learn from.
        y = []
        for i in range(60, len(aapl_data_scaled)):
            X.append(aapl_data_scaled[i-60:i, 0])
            y.append(aapl_data_scaled[i, 0])

        st.success('  Train-Test Split')
        # Split the data into training and testing sets to evaluate the model’s performance properly.
        train_size = int(len(X) * 0.8)
        test_size = len(X) - train_size

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        st.success('  Reshaping Data for LSTM')
        # Finally, we need to reshape our data into a 3D format [samples, time steps, features] required by LSTM layers.
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  

        if 'X_train' not in st.session_state:   st.session_state['X_train'] = ''
        st.session_state['X_train'] = X_train
        if 'y_train' not in st.session_state:   st.session_state['y_train'] = ''
        st.session_state['y_train'] = X_train
        
        st.code(f'X_train {X_train}' )
        st.success("TensorFlow Version: "+ tf.__version__)        

if __name__ == "__main__":
    main()
