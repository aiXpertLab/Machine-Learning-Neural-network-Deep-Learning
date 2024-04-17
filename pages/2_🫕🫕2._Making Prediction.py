import streamlit as st
import numpy as np
from utils import st_def, st_nn

st_def.st_logo(title = "👋 Making Your First Prediction", page_title="Summary",)
st_nn.st_nn2()
# ------------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

def main():
    st.code('''

        # Wrapping the vectors in NumPy arrays
        input_vector = np.array([1.66, 1.56])
        weights_1 = np.array([1.45, -0.66])
        bias = np.array([0.0])

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def make_prediction(input_vector, weights, bias):
            layer_1 = np.dot(input_vector, weights) + bias
            layer_2 = sigmoid(layer_1)
            return layer_2

        prediction = make_prediction(input_vector, weights_1, bias)

        print(f"The prediction result is: {prediction}")

            ''')

    # Wrapping the vectors in NumPy arrays
    input_vector = np.array([1.66, 1.56])
    weights_1 = np.array([1.45, -0.66])
    bias = np.array([0.0])

    prediction = make_prediction(input_vector, weights_1, bias)

    st.write(f"The prediction result is: {prediction}")
    
if __name__ == "__main__":
    main()
