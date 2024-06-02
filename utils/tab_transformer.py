import scipy.special
import streamlit as st
import numpy as np
import scipy

def tf_attention():
    # https://www.analyticsvidhya.com/blog/2023/06/learn-attention-models-from-scratch/
    st.subheader("General Attention")
    
    st.write('1. To begin, we define the word embeddings for a sequence of four words.')
    # encoder representations of four different words
    word_1 = np.array([1, 0, 0])
    word_2 = np.array([0, 1, 0])
    word_3 = np.array([1, 1, 0])
    word_4 = np.array([0, 0, 1])
    
    st.write('2. Next, we generate the weight matrices that will be multiplied with the word embeddings to obtain the queries, keys, and values. For this example, we randomly generate these weight matrices, but in real scenarios, they would be learned during training.')
    np.random.seed(42) 
    W_Q = np.random.randint(3, size=(3, 3))
    W_K = np.random.randint(3, size=(3, 3))
    W_V = np.random.randint(3, size=(3, 3))
    st.write(W_Q)
    st.write(W_K)
    st.write(W_V)
    
    st.write('3. We then calculate the query, key, and value vectors for each word by performing matrix multiplications between the word embeddings and the corresponding weight matrices.')
    query_1 = np.dot(word_1, W_Q)
    key_1 = np.dot(word_1, W_K)
    value_1 = np.dot(word_1, W_V)

    query_2 = np.dot(word_2, W_Q)
    key_2 = np.dot(word_2, W_K)
    value_2 = np.dot(word_2, W_V)

    query_3 = np.dot(word_3, W_Q)
    key_3 = np.dot(word_3, W_K)
    value_3 = np.dot(word_3, W_V)

    query_4 = np.dot(word_4, W_Q)
    key_4 = np.dot(word_4, W_K)
    value_4 = np.dot(word_4, W_V)
    st.write(query_1)
    st.write(key_1)
    st.write(value_1)
    
    st.write('4. Moving on, we score the query vector of the first word against all the key vectors using a dot product operation.')
    scores = np.array([np.dot(query_1,key_1),    np.dot(query_1,key_2),np.dot(query_1,key_3),np.dot(query_1,key_4)])
    st.write(scores)
    
    st.write('5. To generate the weights, we apply the softmax operation to the scores.')
    weights = scipy.special.softmax(scores / np.sqrt(key_1.shape[0]))
    st.write(weights)
    
    st.write('6. Finally, we compute the attention output by taking the weighted sum of all the value vectors.')
    attention=(weights[0]*value_1)+(weights[1]*value_2)+(weights[2]*value_3)+(weights[3]*value_4)
    st.write(attention)
    
def tf_attentioninvector():
    st.subheader("Self-Attention in vector")
    # Representing the encoder representations of four different words
    word_1 = np.array([1, 0, 0])
    word_2 = np.array([0, 1, 0])
    word_3 = np.array([1, 1, 0])
    word_4 = np.array([0, 0, 1])

    # word embeddings.
    words = np.array([word_1, word_2, word_3, word_4])

    # Generating the weight matrices.
    np. random.seed(42)
    W_Q = np. random.randint(3, size=(3, 3))
    W_K = np. random.randint(3, size=(3, 3))
    W_V = np. random.randint(3, size=(3, 3))

    # Generating the queries, keys, and values.
    Q = np.dot(words, W_Q)
    K = np.dot(words, W_K)
    V = np.dot(words, W_V)

    # Scoring vector query.
    scores = np.dot(Q, K.T)

    # Computing the weights by applying a softmax operation.
    weights = scipy.special.softmax(scores / np.sqrt(K.shape[1]), axis=1)

    # Computing the attention by calculating the weighted sum of the value vectors.
    attention = np.dot(weights, V)

    st.write(attention)
