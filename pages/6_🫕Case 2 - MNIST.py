import streamlit as st
from streamlit_extras.stateful_button import button
from utils import st_def, tab_mnist

st_def.st_logo(title = "👋MNIST ", page_title="Summary",)

t1, t2, t3, t4, t5, t6 = st.tabs(["Densely Connected Network/Fully connected/Linear Layer ", "Convnets", "Vision", 'CNN', 'MLP',''])

with t1: tab_mnist.mnist_densely_connected()
with t2: tab_mnist.mnist_convnets()
