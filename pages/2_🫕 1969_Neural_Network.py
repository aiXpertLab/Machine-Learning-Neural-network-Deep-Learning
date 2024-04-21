import streamlit as st, time
from utils import st_def, tab_nn

st_def.st_logo(title = "👋 Neural Network!", page_title="Summary",)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "Wrapping Inputs NumPy", "Prediction", "Train NN","Conclusion"])

with tab1:  tab_nn.nn_general()
with tab2:  tab_nn.nn_wrapping()
with tab3:  tab_nn.nn_prediction()
with tab4:  tab_nn.nn_train()
with tab5:  tab_nn.nn_conclusion()


