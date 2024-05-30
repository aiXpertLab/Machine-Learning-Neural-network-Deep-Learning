import streamlit as st
from streamlit_extras.stateful_button import button
from utils import st_def, tab_dl

st_def.st_logo(title = "👋Transformer 2: Feature Engineering", page_title="Text Cleaning",)
tab_dl.st_dl3()
#------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if button("1. data extract and clean", key="button1"):
    st.text('1. data extract and clean')
    with st.spinner(text="Checking Tensorflow and loading Apple Data  ..."):
        aapl_data = yf.download('IBM', start='2020-01-01', end='2024-01-01')        # Fetch AAPL data
        aapl_data.isnull().sum()        # Checking for missing values
        aapl_data.fillna(method='ffill', inplace=True)        # Filling missing values, if any
    st.code(aapl_data.head())
    
    if button('2. Applying Min-Max Scaling', key="button2"):
        st.text('2. Applying Min-Max Scaling')
        scaler = MinMaxScaler(feature_range=(0,1))      #Applying Min-Max Scaling: This scales the dataset so that all the input features lie between 0 and 1.
        aapl_data_scaled = scaler.fit_transform(aapl_data['Close'].values.reshape(-1,1))
        st.code(aapl_data_scaled[:20])
        #--------------------------------------------------------------------------------------------------------------------------------------
        if button('3. LSTM models require input to be in a sequence format. We transform the data into sequences for the model to learn from.', key="button3"):
            st.text('3. LSTM models require input to be in a sequence format. We transform the data into sequences for the model to learn from.')
            X = []
            y = []
            for i in range(60, len(aapl_data_scaled)):
                X.append(aapl_data_scaled[i-60:i, 0])
                y.append(aapl_data_scaled[i, 0])    
            
            st.code(f'X[:2]= {X[:2]}')
            st.code(f'y[:2]= {y[:20]}')
            
            if button('4. Split the data into training and testing sets.', key="button4"):
                st.text('4. Split the data into training and testing sets.')
                train_size = int(len(X) * 0.8)
                test_size = len(X) - train_size

                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                st.code(X_train[:2])
                st.code(y_train[:2])
                
                if button('5. Finally, reshape data into a 3D format [samples, time steps, features] required by LSTM layers.', key='button5'):
                    st.write('5. Finally, reshape data into a 3D format [samples, time steps, features] required by LSTM layers.')
                    X_train, y_train = np.array(X_train), np.array(y_train)
                    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  

                    st.code(f'X_train {X_train}' )
                    st.success('Transformer Done!')

                    # 3._Creating LSTM Layers.py
                    # st.session_state
                    if button('6. Integraing the Attention Mechanism',key='button6'):

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
                                
                        st.success(f'Check terminal: Our LSTM model:  {model.summary()}')
                        st.success("TensorFlow Version: "+ tf.__version__)        
                        
                        if button('7. Optimizing the Model',key='but7'):
                            from keras.layers import BatchNormalization
                            # Adding Dropout and Batch Normalization
                            model.add(Dropout(0.2))
                            model.add(BatchNormalization())
                            model.summary()

                            st.success('This custom layer computes a weighted sum of the input sequence, allowing the model to pay more attention to certain time steps.')
                            st.success("TensorFlow Version: "+ tf.__version__)        


