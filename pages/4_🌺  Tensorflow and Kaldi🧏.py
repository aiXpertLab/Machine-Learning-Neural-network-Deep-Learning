import streamlit as st, time
from utils import st_def, tab_tf

st_def.st_logo(title = "👋 Tensorflow!", page_title="Summary",)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["General", "Keras Sequential API", "Functional Models API", "Standard Network Models","tf.Dataset" , "Kaldi"])

with tab1:  tab_tf.tf_general()
with tab2:  tab_tf.tf_keras()
with tab3: tab_tf.tf_functional_models()
with tab4: tab_tf.tf_standard_models()
with tab5: tab_tf.tf_dataset()
with tab6: st_def.st_kaldi()
