import streamlit as st

def tf_general():
    contents="""
        ### 🚀 Tensorflow 🍨

        TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. 
        Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms.
        We will use a high-level API named `tf.keras` to define and train machine learning models and to make predictions. 
        tf.keras is the TensorFlow variant of the open-source Keras API.

        ### 📄Key Features📚:
        -  🔍 No Coding Required: Say goodbye to developer fees and lengthy website updates. Store Spark’s user-friendly API ensures a smooth integration process.
        -  📰 Empower Your Business: Offer instant customer support, improve lead generation, and boost conversion rates — all with minimal setup effort.
        -  🍨 Seamless Integration: Maintain your existing website design and user experience. Store Spark seamlessly blends in, providing a unified customer journey.
        """
    st.markdown(contents)
    st.image("./images/tf.png")


def tf_keras():
    st.markdown("""Keras Model Life-Cycle: `Define -> Compile -> Fit -> Evaluate -> Make Predictions.`
***
##### Step 1. Define Network.
Neural networks are defined in Keras as a `sequence of layers`. The container for these layers is the `Sequential class`. Create an instance
of the Sequential class. Then you can create your layers and add them in the order that they should be connected. For example, we can do this in two steps:
```
model = Sequential()
model.add(Dense(2))

or: creating an array of layers, passing to constructor of the Sequential class.

layers = [Dense(2)]
model = Sequential(layers)
```
The first layer in the network must define the number of inputs to expect. The way that this
is specified can differ depending on the network type, but for a Multilayer Perceptron model
this is specified by the input dim attribute. For example, a small Multilayer Perceptron model
with 2 inputs in the visible layer, 5 neurons in the hidden layer and one neuron in the output
layer can be defined as:
```
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(1))
```
Think of a Sequential model as a pipeline with your raw data fed in at the bottom and
predictions that come out at the top. This is a helpful conception in Keras as components
that were traditionally associated with a layer can also be split out and added as separate
layers, clearly showing their role in the transform of data from input to prediction. For example,
activation functions that transform a summed signal from each neuron in a layer can be extracted
and added to the Sequential as a layer-like object called the Activation class.
```
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
```

The choice of activation function is most important for the output layer as it will define the
format that predictions will take. For example, below are some common predictive modeling
problem types and the structure and standard activation function that you can use in the output
layer:
- Regression: Linear activation function, or linear (or None), and the number of neurons
matching the number of outputs.
- Binary Classification (2 class): Logistic activation function, or sigmoid, and one
neuron the output layer.
- Multiclass Classification (>2 class): Softmax activation function, or softmax, and
one output neuron per class value, assuming a one hot encoded output pattern.

***
##### Step 2. Compile Network.

Once we have defined our network, we must compile it. Compilation transforms the simple sequence of layers that we defined into a highly efficient series of matrix
transforms in a format intended to be executed on your GPU or CPU.

Compilation requires a number of parameters to be specified. 
- The optimization algorithm to use to train the network and 
- the loss function used to evaluate the network that is minimized by the optimization algorithm. 

For example, below is a case of compiling a defined model and specifying the `stochastic gradient descent (sgd)` optimization algorithm and the `mean squared error` loss
function, intended for a regression type problem.

```
model.compile(optimizer='sgd', loss='mean_squared_error')

or, if need parameters:
algorithm = SGD(lr=0.1, momentum=0.3)
model.compile(optimizer=algorithm, loss='mean_squared_error')

```
The type of predictive modeling problem imposes constraints on the type of loss function
that can be used. For example, below are some standard loss functions for different predictive
model types:
- Regression: Mean Squared Error or mean squared error.
- Binary Classification (2 class): Logarithmic Loss, also called cross-entropy or binary crossentropy.
- Multiclass Classification (>2 class): Multiclass Logarithmic Loss or categorical crossentropy.

The most common optimization algorithm is `stochastic gradient descent`, but Keras also
supports a suite of other state-of-the-art optimization algorithms that work well with little or
no configuration. Perhaps the most commonly used optimization algorithms because of their
generally better performance are:
- Stochastic Gradient Descent, or sgd, that requires the tuning of a learning rate and
momentum.
- Adam, or adam, that requires the tuning of learning rate.
- RMSprop, or rmsprop, that requires the tuning of learning rate.

Finally, you can also specify metrics to collect while fitting your model in addition to the
loss function. Generally, the most useful additional metric to collect is accuracy for classification
problems. The metrics to collect are specified by name in an array. For example:

`model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])`


***
##### Step 3. Fit Network.

Once the network is compiled, it can be fit, which means adapting the model weights in response to a training dataset. Fitting the network requires the training data to be specified, both 
- a matrix of input patterns, X, 
- and an array of matching output patterns, y. 

The network is trained using the `backpropagation algorithm 反向传播算法，简称BP算法` and optimized according to the optimization algorithm
and loss function specified when compiling the model.

The backpropagation algorithm requires that the network be trained for a specified number of `epochs` or `exposures` to the training dataset. 
Each epoch can be partitioned into groups of input-output `pattern pairs` called `batches`. 
This defines the number of patterns that the network is exposed to before the weights are updated within an epoch. It is also an efficiency
optimization, ensuring that not too many input patterns are loaded into memory at a time. A minimal example of fitting a network is as follows:
`history = model.fit(X, y, batch_size=10, epochs=100)`

Once fit, a history object is returned that provides a summary of the performance of the model during training. 
This includes both the loss and any additional metrics specified when compiling the model, recorded each epoch. 

Training can take a long time, from seconds to hours to days depending on the size of the network and the size of the training data.
By default, a progress bar is displayed on the command line for each epoch. 

***
##### Step 4. Evaluate Network.
Once the network is trained, it can be evaluated. We can evaluate the performance of
the network on a separate dataset, unseen during testing. 
The model evaluates the loss across all of the test patterns, as well as any other metrics
specified when the model was compiled, like classification accuracy. A list of evaluation metrics
is returned. For example, for a model compiled with the accuracy metric, we could evaluate it
on a new dataset as follows:
`loss, accuracy = model.evaluate(X, y)`

***
##### Step 5. Make Predictions.
Once we are satisfied with the performance of our fit model, we can use it to make predictions
on new data. This is as easy as calling the predict() function on the model with an array of
new input patterns. For example:
`predictions = model.predict(X)`

The predictions will be returned in the format provided by the output layer of the network.
In the case of a regression problem, these predictions may be in the format of the problem
directly, provided by a linear activation function. For a binary classification problem, the
predictions may be an array of probabilities for the first class that can be converted to a 1 or 0
by rounding.
For a multiclass classification problem, the results may be in the form of an array of
probabilities (assuming a one hot encoded output variable) that may need to be converted
to a single class output prediction using the argmax() NumPy function. 

Alternately, for
classification problems, we can use the predict classes() function that will automatically
convert the predicted probabilities into class integer values.
`predictions = model.predict_classes(X)        `
        
        """)


def tf_functional_models():
    st.markdown("""
The `sequential API` allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple input or output
layers. 

The `functional API` in Keras is an alternate way of creating models that offers a lot more flexibility, including creating more complex models.
It specifically allows you to define multiple input or output models as well as models that share layers. More than that, it allows you to define ad hoc acyclic network graphs. Models are
defined by creating instances of layers and connecting them directly to each other in pairs, then
defining a Model that specifies the layers to act as the input and output to the model. Let’s
look at the three unique aspects of Keras functional API in turn:

***** 1 Defining Input
Unlike the Sequential model, you must create and define a standalone Input layer that specifies
the shape of input data. The input layer takes a shape argument that is a tuple that indicates the
dimensionality of the input data. When input data is one-dimensional, such as for a Multilayer
Perceptron, the shape must explicitly leave room for the shape of the minibatch size used when
splitting the data when training the network. Therefore, the shape tuple is always defined
with a hanging last dimension (2,), this is the way you must define a one-dimensional tuple in
Python, for example:
`visible = Input(shape=(2,))`

***** 2 Connecting Layers
The layers in the model are connected pairwise. This is done by specifying where the input
comes from when defining each new layer. A bracket or functional notation is used, such that
after the layer is created, the layer from which the input to the current layer comes from is
specified. Let’s make this clear with a short example. We can create the input layer as above,
then create a hidden layer as a Dense that receives input only from the input layer.
```
visible = Input(shape=(2,))
hidden = Dense(2)(visible)
```

Note it is the (visible) layer after the creation of the Dense layer that connects the input
layer’s output as the input to the Dense hidden layer. It is this way of connecting layers pairwise
that gives the functional API its flexibility. For example, you can see how easy it would be to
start defining ad hoc graphs of layers.

***** 3 Creating the Model
After creating all of your model layers and connecting them together, you must define the model.
As with the Sequential API, the model is the thing you can summarize, fit, evaluate, and use to
make predictions. Keras provides a Model class that you can use to create a model from your
created layers. It requires that you only specify the input and output layers. For example:
```
visible = Input(shape=(2,))
hidden = Dense(2)(visible)
model = Model(inputs=visible, outputs=hidden)
```
        """)

def tf_standard_models():
    st.markdown("""

1. Multilayer Perceptron
2. Convolutional Neural Network
3. Recurrent Neural Network
        """)

def tf_dataset():
    import numpy as np
    import tensorflow as tf
    random_numbers = np.random.normal(size = (1000,16))
    dataset = tf.data.Dataset.from_tensor_slices(random_numbers)
    
    for i, element in enumerate(random_numbers):
        st.text(element.shape)
        if i>=2: break
        
    batched_dataset = dataset.batch(32)
    for i, element in enumerate(batched_dataset):
        st.text(element.shape)
        if i>=2: break