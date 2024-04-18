import streamlit as st
from utils import st_def, st_DL

st_def.st_logo(title = "👋Transformer 2: Feature Engineering", page_title="Text Cleaning",)
st_DL.st_dl3()
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
    if 'X_train' not in st.session_state:
        st.info("Click: '2. Transformer - Cleaning and Feature Engineering' first.")
    else:
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        st.code(f'X_train {X_train}')

        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply

        model = Sequential()

        # Adding LSTM layers with return_sequences=True
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50, return_sequences=False))  # Only the last time step
        
        # Adding a Dense layer to match the output shape with y_train
        model.add(Dense(1))

        # Compiling the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)
                
        
        if 'model' not in st.session_state:   
            st.session_state['model'] = ''
        st.session_state['model'] = model

        st.success(f'Our LSTM model:  {model.summary()}')
        st.success("TensorFlow Version: "+ tf.__version__)        

if __name__ == "__main__":
    main()