import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def st_ml1():
    st.markdown("""

1. Numpy, OpenCV, and Scikit are used when working with images
2. NLTK along with Numpy and Scikit again when working with text
3. Librosa for audio applications
4. Matplotlib, Seaborn, and Scikit for data representation
5. TensorFlow and Pytorch for Deep Learning applications
6. Scipy for Scientific Computing
7. Pandas for high-level data structures and analysis
                
                """)
    st.image("./data/images/inputdata.png")

def st_ml2():
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
    st.image("./data/images/inputdata2.png")

def st_ml3():
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
    st.image("./data/images/inputdata3.png")

    st.markdown("""

                #### 🍨Adjusting the Parameters With Backpropagation📰
                In this section, you’ll walk through the backpropagation process step by step, starting with how you update the bias. 
                You want to take the derivative of the error function with respect to the bias, derror_dbias. 
                Then you’ll keep going backward, taking the partial derivatives until you find the bias variable.

                Since you are starting from the end and going backward, you first need to take the partial derivative of the error with respect to the prediction. 
                That’s the derror_dprediction in the image below:
                """)
    st.image("./data/images/inputdata4.png")

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

def st_ml4():
    st.markdown("""
Congratulations! We built a neural network from scratch using NumPy. 
With this knowledge, you’re ready to dive deeper into the world of artificial intelligence in Python.

- What deep learning is and what differentiates it from machine learning
- How to represent vectors with NumPy
- What activation functions are and why they’re used inside a neural network
- What the backpropagation algorithm is and how it works
- How to train a neural network and make predictions

                """)
    st.image("./data/images/nn2.png")

