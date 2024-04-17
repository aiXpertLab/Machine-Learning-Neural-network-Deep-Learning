import streamlit as st
import numpy as np
from utils import st_def, st_nn

st_def.st_logo(title = "👋 Wrapping the Inputs of the Neural Network With NumPy", page_title="Summary",)
st_nn.st_nn1()
# ------------------------------------------------------------------------------------------------------------------------
def main():
    st.code('''
    input_vector = [1.72, 1.23]
    weights_1 = [1.26, 0]
    weights_2 = [2.17, 0.32]

    # Computing the dot product of input_vector and weights_1
    first_indexes_mult = input_vector[0] * weights_1[0]
    second_indexes_mult = input_vector[1] * weights_1[1]
    dot_product_1 = first_indexes_mult + second_indexes_mult
            ''')

    input_vector = [1.72, 1.23]
    weights_1 = [1.26, 0]
    weights_2 = [2.17, 0.32]

    # Computing the dot product of input_vector and weights_1
    first_indexes_mult = input_vector[0] * weights_1[0]
    second_indexes_mult = input_vector[1] * weights_1[1]
    dot_product_1 = first_indexes_mult + second_indexes_mult

    dot_product_2 = np.dot(input_vector, weights_2)

    st.markdown(f"The dot product of input with weights_1 is: {dot_product_1}")
    st.markdown(f"The dot product of input with weights_2 is: {dot_product_2}")
    st.markdown("If the output result can be either 0 or 1. This is a classification problem, a subset of supervised learning problems in which you have a dataset with the inputs and the known targets. ")

if __name__ == "__main__":
    main()

