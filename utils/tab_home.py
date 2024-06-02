import streamlit as st

def home_main_contents():
    st.image('./images/dl/compare.png')
    st.write("""
             
1843年，埃达• 洛夫莱斯伯爵夫人对这项发明评论道：“分析机谈不上能创造什么东西。它只能完成我们命令它做的任何事情……它的职责是帮助我们去
实现我们已知的事情。”

随后，人工智能先驱阿兰• 图灵在其1950 年发表的具有里程碑意义的论文“计算机器和智
能”a 中，引用了上述评论并将其称为“洛夫莱斯伯爵夫人的异议”。图灵在这篇论文中介绍了图
灵测试以及日后人工智能所包含的重要概念。在引述埃达• 洛夫莱斯伯爵夫人的同时，图灵还
思考了这样一个问题：通用计算机是否能够学习与创新？他得出的结论是“能”。
机器学习的概念就来自于图灵的这个问题：对于计算机而言，除了“我们命令它做的任何
事情”之外，它能否自我学习执行特定任务的方法？计算机能否让我们大吃一惊？如果没有程
序员精心编写的数据处理规则，计算机能否通过观察数据自动学会这些规则？
图灵的这个问题引出了一种新的编程范式。在经典的程序设计（即符号主义人工智能的范
式）中，人们输入的是规则（即程序）和需要根据这些规则进行处理的数据，系统输出的是答案
（见图1-2）。利用机器学习，人们输入的是数据和从这些数据中预期得到的答案，系统输出的是
规则。这些规则随后可应用于新的数据，并使计算机自主生成答案。
             
人工智能诞生于20 世纪50 年代，当时计算机科学这一新兴领域的少数先驱开始提出疑问：
计算机是否能够“思考”？我们今天仍在探索这一问题的答案。人工智能的简洁定义如下：努
力将通常由人类完成的智力任务自动化。因此，人工智能是一个综合性的领域，不仅包括机器
学习与深度学习，还包括更多不涉及学习的方法。例如，早期的国际象棋程序仅包含程序员精
心编写的硬编码规则，并不属于机器学习。在相当长的时间内，许多专家相信，只要程序员精
心编写足够多的明确规则来处理知识，就可以实现与人类水平相当的人工智能。这一方法被称
为符号主义人工智能（`symbolic AI`），从20 世纪50 年代到80 年代末是人工智能的主流范式。
在20 世纪80 年代的专家系统（`expert system`）热潮中，这一方法的热度达到了顶峰。

虽然符号主义人工智能适合用来解决定义明确的逻辑问题，比如下国际象棋，但它难以给
出明确的规则来解决更加复杂、模糊的问题，比如图像分类、语音识别和语言翻译。于是出现
了一种新的方法来替代符号主义人工智能，这就是机器学习（`machine learning`）。
             
             
             """)
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

Think of AI as the entire field of transportation.
Machine learning would be like cars in general.
Deep learning would be a specific type of car, like a sports car.
Neural networks would be the engine of the car.
Large language models would be a specific type of car designed for racing, built on a powerful engine.

            """
    st.markdown(main_contents)
    st.image("./images/aihistory.png")
    



def st_text_preprocessing_contents():
    st.markdown("""
        - Normalize Text
        - Remove Unicode Characters
        - Remove Stopwords
        - Perform Stemming and Lemmatization
    """)    

def st_load_ML():
    st.image("./images/mlpipeline.png")


def st_kaldi():
    contents="""
        ### 🚀 Kaldi 🍨
        
        A paper is stored in data.
        https://eleanorchodroff.com/tutorial/kaldi/training-overview.html
        

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
    st.image("./images/kaldi.png")

def home_dev():
    st.markdown('### Python')
    import scipy, numpy, matplotlib, pandas, statsmodels, sklearn, PIL
    st.text('scipy: %s' % scipy.__version__)
    st.text('numpy: %s' % numpy.__version__)
    st.text('matplotlib: %s' % matplotlib.__version__)
    st.text('pandas: %s' % pandas.__version__)
    st.text('statsmodels: %s' % statsmodels.__version__)
    st.text('sklearn: %s' % sklearn.__version__)
    st.text(f'PIL: {PIL.__version__}' )
    
    st.markdown('### Deep Learning')
    import tensorflow, keras
    st.text('TF: %s' % tensorflow.__version__)
    st.text('Keras: %s' % keras.__version__)
