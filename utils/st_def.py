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
        st.image("./data/images/ai.png")
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
        st.image("./data/images/aihistory.png")
    
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
    st.image("./data/images/MachineLearning.png")
    st.markdown("""
Supervised learning models can make predictions after seeing lots of data with the correct answers and then discovering the connections between the elements in the data that produce the correct answers. This is like a student learning new material by studying old exams that contain both questions and answers. Once the student has trained on enough old exams, the student is well prepared to take a new exam. 
These ML systems are "supervised" in the sense that a human gives the ML system data with the known correct results.
Two of the most common use cases for supervised learning are `regression` and `classification`.

**Unsupervised** learning models make predictions by being given data that does not contain any correct answers. 
An unsupervised learning model's goal is to identify meaningful `patterns` among the data. 
In other words, the model has no hints on how to categorize each piece of data, but instead it must infer its own rules.

A commonly used unsupervised learning model employs a technique called `clustering`. The model finds data points that demarcate natural groupings.

Under supervised ML, two major subcategories are:

- Regression machine learning systems – Systems where the value being predicted falls somewhere on a continuous spectrum. These systems help us with questions of “How much?” or “How many?”
- Classification machine learning systems – Systems where we seek a yes-or-no prediction, such as “Is this tumor cancerous?”, “Does this cookie meet our quality standards?”, and so on.

**Unsupervised machine learning** is typically tasked with finding relationships within data. There are no training examples used in this process. Instead, the system is given a set of data and tasked with finding patterns and correlations therein. A good example is identifying close-knit groups of friends in social network data.

The machine learning algorithms used to do this are very different from those used for supervised learning, and the topic merits its own post. However, for something to chew on in the meantime, take a look at clustering algorithms such as k-means, and also look into dimensionality reduction systems such as principle component analysis. You can also read our article on semi-supervised image classification.

Deep learning is a subset of machine learning, so it doesn't replace traditional machine learning techniques but rather complements them. While deep learning has shown remarkable success in various tasks such as image recognition, natural language processing, and speech recognition, there are still many scenarios where traditional machine learning algorithms excel.

Machine learning encompasses a broad range of techniques beyond deep learning, including:

1. Supervised Learning: Deep learning is just one approach to supervised learning. Traditional machine learning algorithms like decision trees, support vector machines, and random forests are still widely used for tasks where interpretability and transparency are important, or when the dataset is not large enough to benefit from deep learning's complexity.
2. Unsupervised Learning: Techniques like clustering, dimensionality reduction, and association rule learning are essential in situations where labeled data is scarce or unavailable. Deep learning models typically require large amounts of labeled data for training, which may not always be feasible.
3. Semi-Supervised Learning: This approach leverages both labeled and unlabeled data, which is common in real-world scenarios. Traditional machine learning algorithms, along with some recent advancements, play a crucial role in semi-supervised learning.
4. Feature Engineering: Crafting relevant features from raw data is a crucial step in building effective machine learning models. While deep learning models can automatically learn features from raw data, feature engineering is still relevant and necessary in many cases to improve model performance.
5. Interpretability and Explainability: Understanding why a model makes certain predictions is crucial in many applications, such as healthcare and finance. Traditional machine learning algorithms often offer more transparency and interpretability compared to deep learning models, making them preferable in certain scenarios.
6. Computational Efficiency: Deep learning models, especially large neural networks, can be computationally expensive to train and deploy. Traditional machine learning algorithms are often more computationally efficient and can be deployed on resource-constrained devices.

In summary, while deep learning has revolutionized many fields, traditional machine learning techniques remain essential in various working environments due to their interpretability, efficiency, and effectiveness in scenarios with limited data or computational resources.

**Machine learning's goal is to predict well on new data drawn from a (hidden) true probability distribution**. 
""")    
    st.image("./data/images/clustering.png")

def st_DL():
    st.markdown("""
Deep learning is a technique used to make predictions using data, and it heavily relies on neural networks. 
 
Deep learning framework like **TensorFlow** or **PyTorch** instead of building your own neural network. 
That said, having some knowledge of how neural networks work is helpful because you can use it to better architect your deep learning models.

**Traditional Machine Learning:**

- These models typically involve feature engineering, where domain-specific features are manually crafted from raw data to feed into the learning algorithm.
- Examples of traditional machine learning algorithms include linear regression, logistic regression, decision trees, support vector machines, and k-nearest neighbors, among others.
- While some traditional machine learning algorithms may use ensemble techniques that combine multiple models (e.g., random forests, gradient boosting), they are not typically referred to as "multi-layer" in the same sense as deep neural networks.

**Deep Learning:**

Deep learning, on the other hand, specifically refers to neural networks with multiple layers (hence the term "deep").
- Deep learning architectures consist of multiple layers of interconnected neurons, allowing them to learn complex representations and hierarchies of features directly from raw data.
- Deep learning models are capable of automatically learning feature representations from data without requiring explicit feature engineering.
- Examples of deep learning architectures include convolutional neural networks (CNNs) for image analysis, recurrent neural networks (RNNs) for sequential data, and transformer-based architectures for natural language processing.
- The depth of neural networks in deep learning refers to the number of layers, and deep networks may consist of dozens or even hundreds of layers.
""")    
    st.image("./data/images/book.png")

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
    st.image("./data/images/nn1.png")

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
    st.image("./data/images/tf.png")

