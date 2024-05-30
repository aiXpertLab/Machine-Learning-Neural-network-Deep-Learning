import streamlit as st
from streamlit_extras.stateful_button import button

def cp_general():
    st.markdown('''
Convolution and the convolutional layer are the major building blocks used in convolutional
neural networks. 

A convolution is the simple application of a filter to an input that results in an
activation. 

Repeated application of the same filter to an input results in a map of activations
called a feature map, indicating the locations and strength of a detected feature in an input,
such as an image. 

The innovation of convolutional neural networks is the ability to automatically
learn a large number of filters in parallel specific to a training dataset under the constraints of
a specific predictive modeling problem, such as image classification. The result is that highly
specific features can be detected anywhere on input images. In this tutorial, you will discover
how convolutions work in the convolutional neural network. After completing this tutorial, you
will know:

- Convolutional neural networks apply a filter to an input to create a feature map that summarizes the presence of detected features in the input.
- Filters can be handcrafted, such as line detectors, but the innovation of convolutional neural networks is to learn the filters during training in the context of a specific prediction problem.
- How to calculate the feature map for one- and two-dimensional convolutional layers in a convolutional neural network.    
    
    ''')


def cp_convolution_cnn():
    st.image("./images/zhang1.gif")
    st.write('''
             
The convolutional neural network, or CNN for short, is a specialized type of neural network
model designed for working with `two-dimensional image` data, although they can be used with one-dimensional and three-dimensional data. 

Central to the convolutional neural network is the `convolutional layer` that gives the network its name. This layer performs an operation called a convolution. 

In the context of a convolutional neural network, a convolution is a linear operation that involves the multiplication of a set of weights with the input, 
much like a traditional neural network. 

Given that the technique was designed for two-dimensional input, the multiplication is performed between an array of input data and a two-dimensional 
array of weights, called a filter or a kernel.

The filter is smaller than the input data and the type of multiplication applied between a
filter-sized patch of the input and the filter is a `dot product`. 

A dot product is the `element-wise` multiplication between the filter-sized patch of the input and filter, which is then summed,
always resulting in a single value. Because it results in a single value, the operation is often referred to as the scalar product. 

Using a filter smaller than the input is intentional as it allows the same filter (set of weights) to be multiplied by the input array
multiple times at different locations on the input. Specifically, the filter is applied systematically to each overlapping
filter-sized patch of the input data, left to right, top to bottom. This systematic application of the same filter across an image is
a powerful idea. If the filter is designed to detect a specific type of feature in the input, then the application of that filter
systematically across the entire input image allows the filter an opportunity to discover that feature anywhere in the image. This capability is commonly referred to as translation invariance,
e.g. the general interest in whether the feature is present rather than where it was present.
Invariance to local translation can be a very useful property if we care more about whether some feature is present than exactly where it is. For example, when
determining whether an image contains a face, we need not know the location of the eyes with pixel-perfect accuracy, we just need to know that there is an eye on the
left side of the face and an eye on the right side of the face.

The output from multiplying the filter with a filter-sized-patch of the input array one time is a single value. As the filter is applied systematically across the input array, the result is a
two-dimensional array comprised of output values that represent a filtering of the input. As such, the two-dimensional output array from this operation is called a feature map. Once a
feature map is created, we can pass each value in the feature map through a nonlinearity, such as a ReLU, much like we do for the outputs of a fully connected layer.
             
If you come from a digital signal processing field or related area of mathematics, you may understand the convolution operation on a matrix as something different. Specifically, the filter
(kernel) is flipped prior to being applied to the input. Technically, the convolution as described in the use of convolutional neural networks is actually a `cross-correlation`. Nevertheless, in deep
learning, it is referred to as a convolution operation. Many machine learning libraries implement cross-correlation but call it convolution.             
             
             ''')


def cp_convolutional_layers():
    st.image("./images/zhang2.gif")
    if button("1. 1D Convolutional Layer", key="button1"):
        st.markdown("""[0, 0, 0, 1, 1, 0, 0, 0]
        The input to Keras must be three dimensional for a 1D convolutional layer. 

- The first dimension refers to each input sample; in this case, we only have one sample. 
- The second dimension refers to the length of each sample; in this case, the length is eight. 
- The third dimension refers to the number of channels in each sample; in this case, we only have a single channel. 
        
Therefore, the shape of the input array will be [1, 8, 1].
        
        #1 define input data
        data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
        data = data.reshape(1, 8, 1)
        
We will define a model that expects input samples to have the shape [8, 1]. The model will have a single filter with the shape of 3,
        or three elements wide. Keras refers to the shape of the filter as the kernel size (the required second argument to the layer).
        
        #2 create model
        model = Sequential()
        model.add(Conv1D(1, 3, input_shape=(8, 1)))
                    

By default, the filters in a convolutional layer are initialized with random weights. In this contrived example, we will manually specify the weights for the single filter. We will define a
filter that is capable of detecting bumps, that is a high input value surrounded by low input values, as we defined in our input example. The three element filter we will define looks as
follows: [0, 1, 0]

Each filter also has a bias input value that also requires a weight that we will set to zero.
Therefore, we can force the weights of our one-dimensional convolutional layer to use our handcrafted filter as follows:

        #3 define a vertical line detector
        weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
        #4 store the weights in the model
        model.set_weights(weights)

The weights must be specified in a three-dimensional structure, in terms of rows, columns,
and channels. The filter has a single row, three columns, and one channel. We can retrieve the weights and confirm that they were set correctly.

        #5 confirm they were stored
        print(model.get_weights())

    Finally, we can apply the single filter to our input data. We can achieve this by calling the
    predict() function on the model. This will return the feature map directly: that is the output
    of applying the filter systematically across the input sequence.

        #6 apply filter to input data
        yhat = model.predict(data)
        print(yhat)
        """)    
        
        # example of calculation 1d convolutions
        from numpy import asarray
        from keras.models import Sequential
        from keras.layers import Conv1D
        # define input data
        data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
        data = data.reshape(1, 8, 1)
        # create model
        model = Sequential()
        model.add(Conv1D(1, 3, input_shape=(8, 1)))
        # define a vertical line detector
        weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]
        # store the weights in the model
        model.set_weights(weights)
        # confirm they were stored
        st.write(model.get_weights())
        # apply filter to input data
        yhat = model.predict(data)
        st.write(yhat)
        
        st.divider()
        st.subheader("2. 2D Convolutional Layer")
        if button("2. 2D Convolutional Layer", key="button2"):
            from numpy import asarray
            from keras.models import Sequential
            from keras.layers import Conv2D
            # define input data
            data = [[0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0]]
            data = asarray(data)
            data = data.reshape(1, 8, 8, 1)
            # create model
            model = Sequential()
            model.add(Conv2D(1, (3,3), input_shape=(8, 8, 1)))
            # define a vertical line detector
            detector = [[[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]],
            [[[0]],[[1]],[[0]]]]
            weights = [asarray(detector), asarray([0.0])]
            # store the weights in the model
            model.set_weights(weights)
            # confirm they were stored
            print(model.get_weights())
            # apply filter to input data
            yhat = model.predict(data)
            for r in range(yhat.shape[1]):
                # print each column in the row
                st.write([yhat[0,r,c,0] for c in range(yhat.shape[2])])
            
            
            
        
