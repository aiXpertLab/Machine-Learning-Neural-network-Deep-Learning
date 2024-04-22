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


def nn_train():
    st.markdown("""
            1. Taking the input data
            2. Making a prediction`
            3. `Comparing the prediction to the desired output`
            4. `Adjusting its internal state to predict correctly the next time`

            In the process of training the neural network, you first assess the error and then adjust the weights accordingly. 
            To adjust the weights, you’ll use the `gradient descent` and `backpropagation algorithms`. 
            Gradient descent is applied to find the direction and the rate to update the parameters.

            Before making any changes in the network, you need to compute the error. That’s what you’ll do in the next section.

            #### 🍨Computing the Prediction Error
            To understand the magnitude of the error, you need to choose a way to measure it. 
            The function used to measure the error is called the `cost function`, or `loss function`. 
            We’ll use the `mean squared error (MSE)` as the cost function. 

            The network can make a mistake by outputting a value that’s higher or lower than the correct value. 
            Since the MSE is the squared difference between the prediction and the correct result, with this metric you’ll always end up with a positive value.

            One implication of multiplying the difference by itself is that bigger errors have an even larger impact, and smaller errors keep getting smaller as they decrease.

            #### 📰Reduce the Error
            The goal is to change the weights and bias variables so you can reduce the error. 
            To understand how this works, you’ll change only the weights variable and leave the bias fixed for now. 
            You can also get rid of the sigmoid function and use only the result of layer_1. All that’s left is to figure out how you can modify the weights so that the error goes down.

            #### 🚀Applying the Chain Rule
            In your neural network, you need to update both the weights and the bias vectors. 
            The function you’re using to measure the error depends on two independent variables, the weights and the bias. 
            Since the weights and the bias are independent variables, you can change and adjust them to get the result you want.

            The network you’re building has two layers, and since each layer has its own functions, you’re dealing with a function composition. 
            This means that the error function is still np.square(x), but now x is the result of another function.

            To restate the problem, now you want to know how to change weights_1 and bias to reduce the error. 
            You already saw that you can use derivatives for this, but instead of a function with only a sum inside, now you have a function that produces its result using other functions.

            Since now you have this function composition, to take the derivative of the error concerning the parameters, you’ll need to use the chain rule from calculus. 
            With the chain rule, you take the partial derivatives of each function, evaluate them, and multiply all the partial derivatives to get the derivative you want.

            Now you can start updating the weights. You want to know how to change the weights to decrease the error. 
            This implies that you need to compute the derivative of the error with respect to weights. 
            Since the error is computed by combining different functions, you need to take the partial derivatives of these functions.
                """)
    st.image("./images/inputdata3.png")

    st.markdown("""

                #### 🍨Adjusting the Parameters With Backpropagation📰
                In this section, you’ll walk through the backpropagation process step by step, starting with how you update the bias. 
                You want to take the derivative of the error function with respect to the bias, derror_dbias. 
                Then you’ll keep going backward, taking the partial derivatives until you find the bias variable.

                Since you are starting from the end and going backward, you first need to take the partial derivative of the error with respect to the prediction. 
                That’s the derror_dprediction in the image below:
                """)
    st.image("./images/inputdata4.png")

    st.markdown("""
            #### 📚Creating the Neural Network Class📄
            Now you know how to write the expressions to update both the weights and the bias. 
            It’s time to create a class for the neural network. Classes are the main building blocks of object-oriented programming (OOP). 
            The NeuralNetwork class generates random start values for the weights and bias variables.

            When instantiating a NeuralNetwork object, you need to pass the learning_rate parameter. 
            You’ll use predict() to make a prediction. The methods _compute_derivatives() and _update_parameters() have the computations you learned in this section. 
                """)
    st.code('''

        class NeuralNetwork:
            def __init__(self, learning_rate):
                self.weights = np.array([np.random.randn(), np.random.randn()])
                self.bias = np.random.randn()
                self.learning_rate = learning_rate

            def _sigmoid(self, x):
                return 1 / (1 + np.exp(-x))

            def _sigmoid_deriv(self, x):
                return self._sigmoid(x) * (1 - self._sigmoid(x))

            def predict(self, input_vector):
                layer_1 = np.dot(input_vector, self.weights) + self.bias
                layer_2 = self._sigmoid(layer_1)
                prediction = layer_2
                return prediction

            def _compute_gradients(self, input_vector, target):
                layer_1 = np.dot(input_vector, self.weights) + self.bias
                layer_2 = self._sigmoid(layer_1)
                prediction = layer_2

                derror_dprediction = 2 * (prediction - target)
                dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
                dlayer1_dbias = 1
                dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

                derror_dbias = (
                    derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
                )
                derror_dweights = (
                    derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
                )

                return derror_dbias, derror_dweights

            def _update_parameters(self, derror_dbias, derror_dweights):
                self.bias = self.bias - (derror_dbias * self.learning_rate)
                self.weights = self.weights - (
                    derror_dweights * self.learning_rate
                )
            ''')

    st.markdown("""

        #### 🔍Training the Network With More Data🍨
        You’ve already adjusted the weights and the bias for one data instance, but the goal is to make the network generalize over an entire dataset. 
        Stochastic gradient descent is a technique in which, at every iteration, the model makes a prediction based on a randomly selected piece of training data, calculates the error, and updates the parameters.

Now it’s time to create the train() method of your NeuralNetwork class. 
You’ll save the error over all data points every 100 iterations because you want to plot a chart showing how this metric changes as the number of iterations increases. 
This is the final train() method of your neural network:
            """)
    st.code("""
class NeuralNetwork:
    # ...

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

            
    """)
    st.markdown("""
    
    In short, you pick a random instance from the dataset, compute the gradients, and update the weights and the bias. 
    You also compute the cumulative error every 100 iterations and save those results in an array. 
    You’ll plot this array to visualize how the error changes during the training process.
    
        """)

def nn_conclusion():
        st.markdown("""
    Congratulations! We built a neural network from scratch using NumPy. 
    With this knowledge, you’re ready to dive deeper into the world of artificial intelligence in Python.

    - What deep learning is and what differentiates it from machine learning
    - How to represent vectors with NumPy
    - What activation functions are and why they’re used inside a neural network
    - What the backpropagation algorithm is and how it works
    - How to train a neural network and make predictions

                    """)
        st.image("./images/nn2.png")


