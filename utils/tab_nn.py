import streamlit as st, numpy as np
from streamlit_extras.add_vertical_space import add_vertical_space

def nn_general():
    st.markdown("""
A neural network is a system that learns how to make predictions by following these steps:

- Taking the input data
- Making a prediction
- Comparing the prediction to the desired output
- Adjusting its internal state to predict correctly the next time

Vectors, layers, and linear regression are some of the building blocks of neural networks. 
The data is stored as vectors, and with Python you store these vectors in arrays. 
Each layer transforms the data that comes from the previous layer. 
You can think of each layer as a feature engineering step, because each layer extracts some representation of the data that came previously.

One cool thing about neural network layers is that the same computations can extract information from any kind of data. 
This means that it doesn’t matter if you’re using image data or text data. 
The process to extract meaningful information and train the deep learning model is the same for both scenarios.

#### 🍨The Process to Train a Neural Network
Training a neural network is similar to the process of trial and error. 
Imagine you’re playing darts for the first time. 
In your first throw, you try to hit the central point of the dartboard. 
Usually, the first shot is just to get a sense of how the height and speed of your hand affect the result. 
If you see the dart is higher than the central point, then you adjust your hand to throw it a little lower, and so on.

With neural networks, the process is very similar: you start with some random weights and bias vectors, make a prediction, compare it to the desired output, and adjust the vectors to predict more accurately the next time. 
The process continues until the difference between the prediction and the correct targets is minimal.

Knowing when to stop the training and what accuracy target to set is an important aspect of training neural networks, mainly because of overfitting and underfitting scenarios.

#### 📰Vectors and Weights
Working with neural networks consists of doing operations with vectors. You represent the vectors as multidimensional arrays. 
Vectors are useful in deep learning mainly because of one particular operation: the dot product. 
The `dot product` of two vectors tells you how similar they are in terms of direction and is scaled by the magnitude of the two vectors.

The main vectors inside a neural network are the weights and bias vectors. 
Loosely, what you want your neural network to do is to check if an input is similar to other inputs it’s already seen. 
If the new input is similar to previously seen inputs, then the outputs will also be similar. That’s how you get the result of a prediction.


#### 🚀The Linear Regression Model
Regression is used when you need to estimate the relationship between a dependent variable and two or more independent variables. Linear regression is a method applied when you approximate the relationship between the variables as linear.

                
                """)
    st.image("./images/nn1.png")

def nn_wrapping():
    st.markdown("""
            A neural network is a system that learns how to make predictions by following these steps:

            1. `Taking the input data`
            2. Making a prediction
            3. Comparing the prediction to the desired output
            4. Adjusting its internal state to predict correctly the next time

            The input data is stored as vectors, and you’ll use NumPy to represent the input vectors of the network as arrays. 

            In this first example, you have an input vector and the other two weight vectors. Who's similar?
                
                """)
    st.image("./images/inputdata.png")
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

    main()



def nn_prediction():
    st.markdown("""
        1. Taking the input data
        2. `Making a prediction`
        3. Comparing the prediction to the desired output
        4. Adjusting its internal state to predict correctly the next time

        Let's keep things straightforward and build a network with only two layers. 

        So far, you’ve seen that the only two operations used inside the neural network were the dot product and a sum. Both are linear operations.

        If you add more layers but keep using only linear operations, then adding more layers would have no effect because each layer will always have some correlation with the input of the previous layer. 

        What you want is to find an operation that makes the middle layers sometimes correlate with an input and sometimes not correlate.

        You can achieve this behavior by using nonlinear functions. These nonlinear functions are called `activation functions`. 
        There are many types of activation functions. The `ReLU` (rectified linear unit), for example, is a function that converts all negative numbers to zero. 
        This means that the network can “turn off” a weight if it’s negative, adding nonlinearity.

        The network you’re building will use the sigmoid activation function. You’ll use it in the last layer, layer_2. 

                """)
    st.image("./images/inputdata2.png")
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
        
    main()
    pass
def nn_train():pass
def nn_conclusion():pass

