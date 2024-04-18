import streamlit as st
from utils import st_def, st_DL

st_def.st_logo(title = "👋Prediction using RNN LSTM with the Attention Mechanism in TensorFlow", page_title="Text Cleaning",)
st_DL.st_dl0()
st.header("Data Acquisition from yfinance")
# with st.spinner(text="Checking Tensorflow and loading Apple Data  ..."):
#------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.success("TensorFlow Version: "+ tf.__version__)

if __name__ == "__main__":
    main()
