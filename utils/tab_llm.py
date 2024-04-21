import streamlit as st

def llm_general():
    st.markdown("""
        Deep learning is a technique used to make predictions using data, and it heavily relies on neural networks. 
    """)
    st.image("./images/llmroadmap.png", use_column_width=True)
    


def llm_datacollection():
    st.image("./images/mlpipeline.png")
    st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")
    st.image("./images/datasource.png")


def llm_types():
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

def llm_featureengineering():
    st.image("./images/llm_featureenginerring.png")
    st.markdown("""

                """)

def llm_preprocessing():
    st.image("./images/llm_preprocessing.png")
    st.markdown("""
   
        """)

def llm_python():
    st.markdown("""

            1. Numpy, OpenCV, and Scikit are used when working with images
            2. NLTK along with Numpy and Scikit again when working with text
            3. Librosa for audio applications
            4. Matplotlib, Seaborn, and Scikit for data representation
            5. TensorFlow and Pytorch for Deep Learning applications
            6. Scipy for Scientific Computing
            7. Pandas for high-level data structures and analysis
               
                """)
    st.image("./images/inputdata.png")


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

def llm_chunking():
    st.code("""

Chunking strategies
Question 1: What is chunking and why do we chunk our data?
Question 2: What are factors influences chunk size?
Question 3: What are the different types of chunking methods available?
Question 4: How to find the ideal chunk size?
Embedding Models
Question 1: What are vector embeddings? And what is an embedding model?
Question 2: How embedding model is used in the context of LLM application?
Question 3: What is the difference between embedding short and long content?
Question 4: How to benchmark embedding models on your data?
Question 5: Walk me through the steps of improving the sentence transformer model used for embedding
Internal working of vector DB
Question 1: What is vector DB?
Question 2: How vector DB is different from traditional databases?
Question 3: How does a vector database work?
Question 4: Explain the difference between vector index, vector DB & vector plugins.
Question 5: What are different vector search strategies?
Question 6: How does clustering reduce search space? When does it fail and how can we mitigate these failures?
Question 7: Explain the Random projection index.
Question 8: Explain the Localitysensitive hashing (LHS) indexing method?
Question 9: Explain the product quantization (PQ) indexing method
Question 10: Compare different Vector indexes and given a scenario, which vector index you would use for a project?
Question 11: How would you decide on ideal search similarity metrics for the use case?
Question 12: Explain the different types and challenges associated with filtering in vector DB.
Question 13: How do you determine the best vector database for your needs?
Advanced search algorithms
Question 1: Why it’s important to have very good search
Question 2: What are the architecture patterns for information retrieval & semantic search, and their use cases?
Question 3: How can you achieve efficient and accurate search results in large scale datasets?
Question 4: Explain the keyword-based retrieval method
Question 5: How to fine-tune re-ranking models?
Question 6: Explain most common metric used in information retrieval and when it fails?
Question 7: I have a recommendation system, which metric should I use to evaluate the system?
Question 8: Compare different information retrieval metrics and which one to use when?
Language models internal working
Question 1: Detailed understanding of the concept of selfattention
Question 2: Overcoming the disadvantages of the self-attention mechanism
Question 3: Understanding positional encoding
Question 4: Detailed explanation of Transformer architecture
Question 5: Advantages of using a transformer instead of LSTM.
Question 6: Difference between local attention and global attention
Question 7: Understanding the computational and memory demands of transformers
Question 8: Increasing the context length of an LLM.
Question 9: How to Optimizing transformer architecture for large vocabularies
Question 10: What is a mixture of expert models?
Supervised finetuning of LLM
Question 1: What is finetuning and why it’ s needed in LLM?
Question 2: Which scenario do we need to finetune LLM?
Question 3: How to make the decision of finetuning?
Question 4: How do you create a fine-tuning dataset for Q&A?
Question 5: How do you improve the model to answer only if there is sufficient context for doing so?
Question 6: How to set hyperparameter for fine-tuning
Question 7: How to estimate infra requirements for fine-tuning LLM?
Question 8: How do you finetune LLM on consumer hardware?
Question 9: What are the different categories of the PEFT method?
Question 10: Explain different reparameterized methods for finetuning LLM?
Question 11: What is catastrophic forgetting in the context of LLMs?
Preference Alignment (RLHF/DPO)
Question 1: At which stage you will decide to go for the Preference alignment type of method rather than SFT?
Question 2: Explain Different Preference Alignment Methods?
Question 3: What is RLHF, and how is it used?
Question 4: Explain the reward hacking issue in RLHF.
Evaluation of LLM system
Question 1: How do you evaluate the best LLM model for your use case?
Question 2: How to evaluate the RAG-based system?
Question 3: What are the different metrics that can be used to evaluate LLM
Question 4: Explain the Chain of verification
Hallucination control techniques
Question 1: What are the different forms of hallucinations?
Question 2: How do you control hallucinations at different levels?
Deployment of LLM
Question 1: Why does quantization not decrease the accuracy of LLM?
Agent-based system
Question 1: Explain the basic concepts of an agent and the types of strategies available to implement agents.
Question 2: Why do we need agents and what are some common strategies to implement agents?
Question 3: Explain ReAct prompting with a code example and its advantages
Question 4: Explain Plan and Execute prompting strategy
Question 5: Explain OpenAI functions with code examples
Question 6: Explain the difference between OpenAI functions vs LangChain Agents.
Prompt Hacking
Question 1: What is prompt hacking and why should we bother about it?
Question 2: What are the different types of prompt hacking?
Question 3: What are the different defense tactics from prompt hacking?
Case study & scenario-based Question
Question 1: How to optimize the cost of the overall LLM System?
We can’t give away all our secrets! :)
We’re feeling extra generous, we’re offering a 50% discount! Use the discount code below

Code: LLM50

Code is valid till 30th May 2024.



                """)

def llm_rag():
    st.code("""
Question 1: How to increase accuracy, and reliability & make answers verifiable in LLM?
Question 2: How does Retrieval augmented generation (RAG) work?
Question 3: What are some of the benefits of using the RAG system?
Question 4: What are the architecture patterns you see when you want to customize your LLM with proprietary data?
Question 5: When should I use Fine-tuning instead of RAG?

                """)

def llm_promptengineering():
    st.code("""

Question 1: What is the difference between Predictive/ Discriminative AI and generative AI?
Question 2: What is LLM & how LLMs are trained?
Question 3: What is a token in the language model?
Question 4: How to estimate the cost of running a SaaS-based & Open source LLM model?
Question 5: Explain the Temperature parameter and how to set it.
Question 6: What are different decoding strategies for picking output tokens?
Question 7: What are the different ways you can define stopping criteria in a large language model?
Question 8: How to use stop sequence in LLMs?
Question 9: Explain the basic structure of prompt engineering.
Question 10: Explain the type of prompt engineering
Question 11: Explain In-Context Learning
Question 12: What are some of the aspects to keep in mind while using few-shots prompting?
Question 13: What are certain strategies to write good prompts?
Question 14: What is hallucination & how can it be controlled using prompt engineering?
Question 15: How do I improve the reasoning ability of my LLM through prompt engineering?
Question 16: How to improve LLM reasoning if your COT prompt fails?

""")

