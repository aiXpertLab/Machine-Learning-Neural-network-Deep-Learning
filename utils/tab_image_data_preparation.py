import streamlit as st

def idp_load_keras():
    st.subheader("2. Load with Keras")
    st.markdown('''
                The Keras deep learning library provides a sophisticated API for loading, preparing, and augmenting image data. Also included in the API are some undocumented functions that allow
                you to quickly and easily load, convert, and save image files. These functions can be convenient when getting started on a computer vision deep learning project, allowing you to use the same
                Keras API initially to inspect and handle image data and later to model it.

                The Keras deep learning library provides utilities for working with image data. The main API is the ImageDataGenerator class that combines data loading, preparation, and augmentation.
                We will not cover the ImageDataGenerator class in this tutorial. Instead, we will take a closer look at a few less-documented or undocumented functions that may be useful when working
                with image data and modeling with the Keras API. 

                Specifically, Keras provides functions for loading, converting, and saving image data. The functions are in the `utils.py` function and
                exposed via the image.py module. These functions can be useful convenience functions when getting started on a new deep learning computer vision project or when you need to inspect
                specific images.

                Some of these functions are demonstrated when working with pre-trained models in the Applications section of the API documentation. All image handling in Keras requires that the
                Pillow library is installed. Let’s take a closer look at each of these functions in turn.

                ''')
    
    st.code('''
            # example of loading an image with the Keras API
            from keras.preprocessing.image import load_img
            # load the image
            img = load_img('bondi_beach.jpg')
            # report details about the image
            print(type(img))
            print(img.format)
            print(img.mode)
            print(img.size)
            # show the image
            img.show()
            ''')

    from keras.preprocessing.image import load_img
    # load the image
    img = load_img('./images/zhang.gif')
    # report details about the image
    print(type(img))
    print(img.format)
    print(img.mode)
    print(img.size)
    # show the image
    st.image('./images/zhang.gif')
    


def idp_load_pil():
    st.subheader("1. Load with PIL")
    st.code('''
            # load and show an image with Pillow
            from PIL import Image
            # load the image
            image = Image.open('opera_house.jpg')
            st.text(f'{image.format} {image.mode} {image.size}')
            image.show()
            ''')
    from PIL import Image
    image = Image.open('./images/zhang3.gif')
    st.text(f'{image.format} {image.mode} {image.size}')
    st.image('./images/zhang2.gif')
    
    st.markdown('''
    The Matplotlib wrapper functions can be more effective than using Pillow directly. Nev-
    ertheless, you can access the pixel data from a Pillow Image. Perhaps the simplest way is to
    construct a NumPy array and pass in the Image object. The process can be reversed, converting
    a given array of pixel data into a Pillow Image object using the Image.fromarray() function.
    This can be useful if image data is manipulated as a NumPy array and you then want to save it
    later as a PNG or JPEG file. The example below loads the photo as a Pillow Image object and
    converts it to a NumPy array, then converts it back to an Image object again.
    ```
    from PIL import Image
    from numpy import asarray
    # load the image
    image = Image.open('opera_house.jpg')
    # convert image to numpy array
    data = asarray(image)
    # summarize shape
    st.write(data.shape)
    # create Pillow image
    image2 = Image.fromarray(data)
    # summarize image details
    st.write(image2.format)
    st.write(image2.mode)
    st.write(image2.size)
    
    
    # load all images in a directory
    from os import listdir
    from matplotlib import image
    # load all images in a directory
    loaded_images = list()
    for filename in listdir('images'):
    # load image
    img_data = image.imread('images/' + filename)
    # store loaded image
    loaded_images.append(img_data)
    st.write('> loaded %s %s' % (filename, img_data.shape))
    ```
    ''')

def idp_scale_keras():
    st.subheader("2. Scale with Keras")
    st.markdown('''##### MNIST Handwritten Image Classification Dataset''')
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # load and summarize the MNIST dataset
    from keras.datasets import mnist
    # load dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # summarize dataset shape
    st.write('Train', train_images.shape, train_labels.shape)
    st.write('Test', (test_images.shape, test_labels.shape))
    # summarize pixel values
    st.write('Train', train_images.min(), train_images.max(), train_images.mean(),
    train_images.std())
    st.write('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())
    st.markdown('''
        The ImageDataGenerator class in Keras provides a suite of techniques for scaling pixel values
        in your image dataset prior to modeling. The class will wrap your image dataset, then
        when requested, it will return images in batches to the algorithm during training, validation,
        or evaluation and apply the scaling operations just-in-time. This provides an efficient and
        convenient approach to scaling image data when modeling with neural networks.
        
        The usage of the ImageDataGenerator class is as follows.
        
        1. Load your dataset.
        2. Configure the ImageDataGenerator (e.g. construct an instance).
        3. Calculate image statistics (e.g. call the fit() function).
        4. Use the generator to fit the model (e.g. pass the instance to the fit generator() function).
        5. Use the generator to evaluate the model (e.g. pass the instance to the evaluate generator() function).
        
        The ImageDataGenerator class supports a number of pixel scaling methods, as well as a
        range of data augmentation techniques. We will focus on the pixel scaling techniques in this
        chapter and leave the data augmentation methods to a later discussion. The
        three main types of pixel scaling techniques supported by the ImageDataGenerator class are as follows:
        - Pixel Normalization: scale pixel values to the range 0-1.
        - Pixel Centering: scale pixel values to have a zero mean.
        - Pixel Standardization: scale pixel values to have a zero mean and unit variance.
        
        Pixel standardization is supported at two levels: either per-image (called sample-wise) or
        per-dataset (called feature-wise). Specifically, just the mean, or the mean and standard deviation
        statistics required to standardize pixel values can be calculated from the pixel values in each
        image only (sample-wise) or across the entire training dataset (feature-wise). Other pixel scaling
        methods are supported, such as ZCA, brightening, and more, but we will focus on these three
        most common methods. The choice of pixel scaling is selected by specifying arguments to the
        ImageDataGenerator class when an instance is constructed;
        
        Next, if the chosen scaling method requires that statistics be calculated across the training
        dataset, then these statistics can be calculated and stored by calling the fit() function. When
        evaluating and selecting a model, it is common to calculate these statistics on the training
        dataset and then apply them to the validation and test datasets.
        
        The ImageDataGenerator class can be used to rescale pixel values from the range of 0-255 to the
        range 0-1 preferred for neural network models. Scaling data to the range of 0-1 is traditionally
        referred to as `normalization`. This can be achieved by setting the rescale argument to a ratio by
        which each pixel can be multiplied to achieve the desired range. In this case, the ratio is 1/255 or
        about 0.0039.
    
''')
    
def idp_scale_pil():
    st.subheader("1. Scale with PIL")
    st.code("""
        from PIL import Image
        image = Image.open('./images/idp.png')
        st.text(f'{image.format} {image.mode} {image.size}')        """)
    from PIL import Image
    image = Image.open('./images/idp.png')
    st.text(f'{image.format} {image.mode} {image.size}')       
    
    st.markdown('''
    ##### 1.1 Normalize Pixel Values
    For most image data, the pixel values are integers with values between 0 and 255. 
    Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. 
    As such it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1. 
    
    It is valid for images to have pixel values in the range 0-1 and images can be viewed normally. 
    This can be achieved by dividing all pixels values by the largest pixel value; that is 255.
    This is performed across all channels, regardless of the actual range of pixel values that are present in the image.
    The example below loads the image and converts it into a NumPy array. 
    The data type of the array is reported and the minimum and maximum pixel values across all three channels are then st.writeed. 
    
    Next, the array is converted to the float data type before the pixel values are normalized and the new range of pixel values is reported.
    ''')
    from numpy import asarray
    # load image
    pixels = asarray(image)
    # confirm pixel range is 0-255
    st.write('Data Type: %s' % pixels.dtype)
    st.write('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    st.write(f"Min: {pixels.min():.3f}, Max: {pixels.max():.3f}")
    
    st.markdown('''
    ##### 1.2 Center Pixel Values

        A popular data preparation technique for image data is to subtract the mean value from the pixel values. 
        This approach is called centering, as the distribution of the pixel values is centered
        on the value of zero. Centering can be performed before or after normalization. 

        Centering requires that a mean pixel value be calculated prior to subtracting it from the pixel values. There are multiple ways that the mean can be calculated; for example:
        - Per image.
        - Per minibatch of images (under stochastic gradient descent).
        - Per training dataset.

        The mean can be calculated for all pixels in the image, referred to as a global centering, or
        it can be calculated for each channel in the case of color images, referred to as local centering.
        - Global Centering: Calculating and subtracting the mean pixel value across color channels.
        - Local Centering: Calculating and subtracting the mean pixel value per color channel.

        Per-image global centering is common because it is trivial to implement. Also common is per minibatch global or local centering for the same reason: it is fast and easy to implement.
        In some cases, per-channel means are pre-calculated across an entire training dataset. In this case, the image means must be stored and used both during training and any inference with
        the trained models in the future. For models trained on images centered using these means that may be used for transfer learning on new tasks, it can be beneficial or even required to
        normalize images for the new task using the same means.

        ##### 1.3 Standardize Pixel Values
        The distribution of pixel values often follows a Normal or Gaussian distribution, e.g. bell shape. This distribution may be present per image, per minibatch of images, or across the
        training dataset and globally or per channel. As such, there may be benefit in transforming the distribution of pixel values to be a standard Gaussian: that is both centering the pixel values on
        zero and normalizing the values by the standard deviation. The result is a standard Gaussian of pixel values with a mean of 0.0 and a standard deviation of 1.0.

        As with centering, the operation can be performed per image, per minibatch, and across the entire training dataset, and it can be performed globally across channels or locally per
        channel. Standardization may be preferred to normalization and centering alone and it results in both zero-centered values and small input values, roughly in the range -3 to 3, depending
        on the specifics of the dataset. For consistency of the input data, it may make more sense to standardize images per-channel using statistics calculated per minibatch or across the training
        dataset, if possible. 

        ''')
            
def idp_standard_models():
    st.markdown("""

        """)

def idp_standard_models():
    st.markdown("""

        """)

def idp_standard_models():
    st.markdown("""

        """)

def idp_standard_models():
    st.markdown("""

        """)

def idp_lld():
    st.markdown("""Directory Structure
```
data/
data/train/
data/train/red/
data/train/blue/
data/test/
data/test/red/
data/test/blue/
data/validation/
data/validation/red/
data/validation/blue/

data/train/red/car01.jpg
data/train/red/car02.jpg
data/train/red/car03.jpg
...
data/train/blue/car01.jpg
data/train/blue/car02.jpg
data/train/blue/car03.jpg
```

It is possible to write code to manually load image data and return data ready for modeling.
This would include walking the directory structure for a dataset, loading image data, and
returning the input (pixel arrays) and output (class integer). Thankfully, we don’t need to
write this code. Instead, we can use the `ImageDataGenerator` class provided by Keras. The
main benefit of using this class to load the data is that images are loaded for a single dataset in
batches, meaning that it can be used for loading both small datasets as well as very large image
datasets with thousands or millions of images.

Instead of loading all images into memory, it will load just enough images into memory for
the current and perhaps the next few mini-batches when training and evaluating a deep learning
model. I refer to this as progressive loading (or lazy loading), as the dataset is progressively
loaded from file, retrieving just enough data for what is needed immediately. Two additional
benefits of the using the ImageDataGenerator class is that it can also automatically scale pixel
values of images and it can automatically generate augmented versions of images. We will
leave these topics for discussion in another tutorial (see Chapter 9) and instead focus on how
to use the ImageDataGenerator class to load image data from file. The pattern for using the
ImageDataGenerator class is used as follows:
1. Construct and configure an instance of the ImageDataGenerator class.
2. Retrieve an iterator by calling the flow from directory() function.
3. Use the iterator in the training or evaluation of a model.
        """)

def idp_load():
    idp_load_pil()
    idp_load_keras()

def idp_scale():
    idp_scale_pil()
    idp_scale_keras()
