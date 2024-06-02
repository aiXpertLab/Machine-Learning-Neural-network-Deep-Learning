import streamlit as st
from streamlit_extras.stateful_button import button
from utils import st_def, tab_dogs_cats

st_def.st_logo(title = "👋Dogs and Cats", page_title="Summary",)

t1, t2, t3, t4, t5, t6 = st.tabs(["Preprocessing", "Small Dataset", "VGG16 Feature Extracting", 'VGG16 Fine Tuning', '',''])

with t1: tab_dogs_cats.dc_preprocessing()
with t2: tab_dogs_cats.dc_small_dataset()
with t3: tab_dogs_cats.dc_vgg16_feature_extracting()
with t4: tab_dogs_cats.dc_vgg16_fine_tuning()