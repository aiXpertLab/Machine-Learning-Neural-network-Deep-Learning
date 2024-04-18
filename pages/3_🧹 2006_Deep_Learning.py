import streamlit as st
from utils import st_def, tab_dl
st_def.st_logo(title = "2006 👋 Deep Learning!", page_title="2006 Deep Learning",)

tab1, tab2, tab3 = st.tabs(["General", "Theory", "Data"])

with tab1:  tab_dl.dl_general()
with tab2:  tab_dl.dl_theory()
with tab3:  
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np, pandas as pd
    st.text('1. data extract and clean')
    import yfinance as yf
    with st.spinner(text="Checking Tensorflow and loading Apple Data  ..."):
        aapl_data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')        # Fetch AAPL data
        aapl_data.isnull().sum()        # Checking for missing values
        aapl_data.fillna(method='ffill', inplace=True)        # Filling missing values, if any
        if 'aapl_data' not in st.session_state:   
            st.session_state['aapl_data'] = ''
        st.session_state['aapl_data'] = aapl_data
    st.code(aapl_data.head())
   
    #--------------------------------------------------------------------------------------------------------------------------------------
    st.text('2. Applying Min-Max Scaling')
    scaler = MinMaxScaler(feature_range=(0,1))      #Applying Min-Max Scaling: This scales the dataset so that all the input features lie between 0 and 1.
    aapl_data_scaled = scaler.fit_transform(aapl_data['Close'].values.reshape(-1,1))
    st.code(aapl_data_scaled[:20])

    #--------------------------------------------------------------------------------------------------------------------------------------
    st.text('3. LSTM models require input to be in a sequence format. We transform the data into sequences for the model to learn from.')
    X = []
    y = []
    for i in range(60, len(aapl_data_scaled)):
        X.append(aapl_data_scaled[i-60:i, 0])
        y.append(aapl_data_scaled[i, 0])    
    
    st.code(f'X[:2]= {X[:2]}')
    st.code(f'y[:2]= {y[:20]}')
    
    st.text('4. Split the data into training and testing sets.')
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    st.code(X_train[:2])
    st.code(y_train[:2])
    
    st.write('5. Finally, reshape data into a 3D format [samples, time steps, features] required by LSTM layers.')
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  

    st.code(f'X_train {X_train}' )
    st.success('Transformer Done!')
    st.image("./data/images/mlpipeline.png")
    
