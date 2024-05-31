import streamlit as st
from streamlit_extras.stateful_button import button

def cp_general():
    st.markdown('''
Convolution and the convolutional layer are the major building blocks used in convolutional neural networks. 

A convolution is the simple application of a filter to an input that results in an activation. 

Repeated application of the same filter to an input results in a map of activations
called a feature map, indicating the locations and strength of a detected feature in an input, such as an image. 

The innovation of convolutional neural networks is the ability to automatically learn a large number of filters in parallel specific to a training dataset under the constraints of
a specific predictive modeling problem, such as image classification. The result is that highly specific features can be detected anywhere on input images. In this tutorial, you will discover
how convolutions work in the convolutional neural network. After completing this tutorial, you will know:

- Convolutional neural networks apply a filter to an input to create a feature map that summarizes the presence of detected features in the input.
- Filters can be handcrafted, such as line detectors, but the innovation of convolutional neural networks is to learn the filters during training in the context of a specific prediction problem.
- How to calculate the feature map for one- and two-dimensional convolutional layers in a convolutional neural network.    
    
    https://zhuanlan.zhihu.com/p/635438713
    
比如说老板命令张三干活，张三却到楼下打台球去了，后来被老板发现，他非常气愤，扇了张三一巴掌（注意，这就是`输入信号`，脉冲），

于是张三的脸上会渐渐地（贱贱地）鼓起来一个包，张三的脸就是一个`系统`，而鼓起来的包就是张三的脸对巴掌的`响应`，好，这样就和信号系统建立起来意义对应的联系。

下面还需要一些假设来保证论证的严谨：假定张三的脸是线性时不变系统，也就是说，无论什么时候老板打张三一巴掌，打在张三脸的同一位置（这似乎要求张三的脸足够光滑，
如果张三长了很多青春痘，甚至整个脸皮处处连续处处不可导，那难度太大了，我就无话可说了哈哈），张三的脸上总是会在相同的时间间隔内鼓起来一个相同高度的包来，
并且假定以鼓起来的包的`大小`作为`系统输出`。好了，那么，下面可以进入核心内容——卷积了！

如果张三每天都到地下去打台球，那么老板每天都要扇张三一巴掌，不过当老板打张三一巴掌后，5分钟就消肿了，所以时间长了，张三甚至就适应这种生活了……

如果有一天，老板忍无可忍，以0.5秒的间隔开始不间断的扇张三的过程，这样问题就来了，第一次扇张三鼓起来的包还没消肿，第二个巴掌就来了，张三脸上的包就可能鼓起来两倍高，
老板不断扇张三，脉冲不断作用在张三脸上，效果不断叠加了，这样这些效果就可以求和了，结果就是张三脸上的包的高度随时间变化的一个函数了（注意理解）；

如果老板再狠一点，频率越来越高，以至于都辨别不清时间间隔了，那么，求和就变成积分了。

可以这样理解，在这个过程中的某一固定的时刻，张三的脸上的包的鼓起程度和什么有关呢？和之前每次打张三都有关！但是各次的贡献是不一样的，越早打的巴掌，贡献越小，
所以这就是说，某一时刻的输出是之前很多次输入乘以各自的衰减系数之后的叠加而形成某一点的输出，然后再把不同时刻的输出点放在一起，形成一个函数，这就是卷积。

###### 卷积之后的函数就是张三脸上的包的大小随时间变化的函数。

本来张三的包几分钟就可以消肿，可是如果连续打，几个小时也消不了肿了，这难道不是一种平滑过程么？反映到剑桥大学的公式上，f(a)就是第a个巴掌，
g(x-a)就是第a个巴掌在x时刻的作用程度，乘起来再叠加就ok了，大家说是不是这个道理呢？我想这个例子已经非常形象了，你对卷积有了更加具体深刻的了解了吗？
    
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
            
            
            
        
