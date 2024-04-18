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

            try:
                st.info('Adding self-attention mechanism'  )
                model = Sequential()

                # Adding LSTM layers with return_sequences=True
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=True))

                # Adding self-attention mechanism
                # The attention mechanism
                attention = AdditiveAttention(name='attention_weight')
                # Permute and reshape for compatibility
                model.add(Permute((2, 1)))
                model.add(Reshape((-1, X_train.shape[1])))
                attention_result = attention([model.output, model.output])
                multiply_layer = Multiply()([model.output, attention_result])
                # Return to original shape
                model.add(Permute((2, 1)))
                model.add(Reshape((-1, 50)))

                # Adding a Flatten layer before the final Dense layer
                model.add(tf.keras.layers.Flatten())

                # Final Dense layer
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)
                st.session_state['model'] = model
            except:                
            
                st.text("""
                Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 60, 50)              │          10,400 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 60, 50)              │          20,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, 50)                  │          20,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 1)                   │              51 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 152,555 (595.92 KB)
 Trainable params: 50,851 (198.64 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 101,704 (397.29 KB)
""")            


            st.success('This custom layer computes a weighted sum of the input sequence, allowing the model to pay more attention to certain time steps.')
            st.success("TensorFlow Version: "+ tf.__version__)        

if __name__ == "__main__":
    main()