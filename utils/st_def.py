import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def st_sidebar():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        add_vertical_space(2)
        st.write('Made with ❤️ by [aiXpertLab](https://hypech.com)')

    return openai_api_key

def st_main_contents():
        st.image("./images/ai.png")
        main_contents="""
In basic terms, the goal of using AI is to make computers think as humans do. 
For example, to solve a sudoku problem, you can:
- Using Python to write conditional statements and check the constraints to see if you can place a number in each position. 
- Machine learning (ML) and deep learning (DL) are also approaches to solving problems.

The difference between these techniques and a Python script is that ML and DL use training data instead of hard-coded rules, but all of them can be used to solve problems using AI. 

#### 📚Machine Learning
Machine learning is a technique in which you train the system to solve a problem instead of explicitly programming the rules. 
Getting back to the sudoku example in the previous section, to solve the problem using machine learning, you would gather data from solved sudoku games and train a statistical model. 
Statistical models are mathematically formalized ways to approximate the behavior of a phenomenon.

A common machine learning task is **supervised learning**, in which you have a dataset with inputs and known outputs. 
The task is to use this dataset to train a model that predicts the correct outputs based on the inputs. 

The goal of supervised learning tasks is to make predictions for new, unseen data. 
To do that, you assume that this unseen data follows a probability distribution similar to the distribution of the training dataset. 
If in the future this distribution changes, then you need to train your model again using the new training dataset.

#### 📄Features Enginerring
Another name for input data is `feature`, and feature engineering is the process of extracting features from raw data. 

Prediction problems become harder when you use different kinds of data as inputs. 
What if you want to train a model to predict the sentiment in a sentence? 
Or what if you have an image, and you want to know whether it depicts a cat?


An example of a feature engineering technique is `lemmatization`, in which you remove the inflection from words in a sentence. 
If you’re using arrays to store each word of a corpus, then by applying lemmatization, you end up with a less-sparse matrix. 
This can increase the performance of some machine learning algorithms. Creating features using a `bag-of-words model`.

#### 🔍Deep Learning
Deep learning is a technique in which you let the neural network figure out by itself which features are important instead of applying feature engineering techniques. 
This means that, with deep learning, you can bypass the feature engineering process.

Not having to deal with feature engineering is good because the process gets harder as the datasets become more complex. 
For example, how would you extract the data to predict the mood of a person given a picture of her face? 
With neural networks, you don’t need to worry about it because the networks can learn the features by themselves. 
            """
        st.markdown(main_contents)
        st.image("./images/aihistory.png")
    
def st_logo(title="aiXpert!", page_title="Aritificial Intelligence"):
    st.set_page_config(page_title,  page_icon="🚀",)
    st.title(title)

    st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            background-image: url(https://hypech.com/storespark/images/logohigh.png);
            background-repeat: no-repeat;
            padding-top: 80px;
            background-position: 15px 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def st_text_preprocessing_contents():
    st.markdown("""
        - Normalize Text
        - Remove Unicode Characters
        - Remove Stopwords
        - Perform Stemming and Lemmatization
    """)    

def st_load_ML():
    st.image("./images/mlpipeline.png")

def st_nn():
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

def st_tf():
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

def st_kaldi():
    contents="""
        ### 🚀 Kaldi 🍨

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
    st.image("./images/kalditf.png")

