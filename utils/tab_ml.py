import streamlit as st

general="""
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
    """


def ml_datacollection():
    st.image("./images/mlpipeline.png")
    st.markdown(general)    
    st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")
    st.image("./images/datasource.png")


def ml_types():
    st.image("./images/MachineLearning.png")
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
    st.image("./images/clustering.png")

def ml_featureengineering():
    st.image("./images/ml_featureenginerring.png")
    st.markdown("""

                """)

def ml_preprocessing():
    st.image("./images/ml_preprocessing.png")
    st.markdown("""
   
        """)

def st_dl4():
    st.markdown("""
        The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)
def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl0():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def st_dl6():
    st.markdown("""

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_3 (LSTM)               (None, 60, 50)            10400     
                                                                 
 lstm_4 (LSTM)               (None, 60, 50)            20200     
                                                                 
 permute (Permute)           (None, 50, 60)            0         
                                                                 
 reshape (Reshape)           (None, 50, 60)            0         
                                                                 
 permute_1 (Permute)         (None, 60, 50)            0         
                                                                 
 reshape_1 (Reshape)         (None, 60, 50)            0         
                                                                 
 flatten (Flatten)           (None, 3000)              0         
                                                                 
 dense_1 (Dense)             (None, 1)                 3001      
                                                                 
 dropout (Dropout)           (None, 1)                 0         
                                                                 
 batch_normalization (Batch  (None, 1)                 4         
 Normalization)                                                  
                                                                 
=================================================================
Total params: 33605 (131.27 KB)
Trainable params: 33603 (131.26 KB)
Non-trainable params: 2 (8.00 Byte)
_________________________________________________________________


                """)

def st_dl11():
    st.markdown("""
In this guide, we explored the complex yet fascinating task of using LSTM networks with an attention mechanism for stock price prediction, 
specifically for Apple Inc. (AAPL). Key points include:

- LSTM’s ability to capture long-term dependencies in time-series data.
- The added advantage of the attention mechanism in focusing on relevant data points.
- The detailed process of building, training, and evaluating the LSTM model.

#### While LSTM models with attention are powerful, they have limitations:
- The assumption that historical patterns will repeat in similar ways can be problematic, especially in volatile markets.
- External factors like market news and global events, not captured in historical price data, can significantly influence stock prices.
""")

