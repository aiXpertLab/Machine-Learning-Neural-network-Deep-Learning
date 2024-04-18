import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def st_dl0():
    st.markdown("""

LSTM networks are a type of Recurrent Neural Network (RNN) specially designed to remember and process sequences of data over long periods. 
What sets LSTMs apart from traditional RNNs is their ability to preserve information for long durations, courtesy of their unique structure comprising three gates: the input, forget, and output gates.
These gates collaboratively manage the flow of information, deciding what to retain and what to discard, thereby mitigating the issue of vanishing gradients — a common problem in standard RNNs.
    
    """)
    st.image("./data/images/lstm.png")
    
    st.header("Attention Mechanism: Enhancing LSTM")
    st.markdown("""
The attention mechanism, initially popularized in the field of natural language processing, has found its way into various other domains, including finance. 
It operates on a simple yet profound concept: not all parts of the input sequence are equally important. 
By allowing the model to focus on specific parts of the input sequence while ignoring others, the attention mechanism enhances the model’s context understanding capabilities.

Incorporating attention into LSTM networks results in a more focused and context-aware model. 
When predicting stock prices, certain historical data points may be more relevant than others. 
The attention mechanism empowers the LSTM to weigh these points more heavily, leading to more accurate and nuanced predictions.
    """)
    
def st_dl1():
    st.image("./data/images/mlpipeline.png")
    st.markdown("""

LSTM networks are a type of Recurrent Neural Network (RNN) specially designed to remember and process sequences of data over long periods. 
What sets LSTMs apart from traditional RNNs is their ability to preserve information for long durations, courtesy of their unique structure comprising three gates: the input, forget, and output gates.
These gates collaboratively manage the flow of information, deciding what to retain and what to discard, thereby mitigating the issue of vanishing gradients — a common problem in standard RNNs.
    
    """)
    
    st.header("Attention Mechanism: Enhancing LSTM")
    st.markdown("""
The attention mechanism, initially popularized in the field of natural language processing, has found its way into various other domains, including finance. 
It operates on a simple yet profound concept: not all parts of the input sequence are equally important. 
By allowing the model to focus on specific parts of the input sequence while ignoring others, the attention mechanism enhances the model’s context understanding capabilities.

Incorporating attention into LSTM networks results in a more focused and context-aware model. 
When predicting stock prices, certain historical data points may be more relevant than others. 
The attention mechanism empowers the LSTM to weigh these points more heavily, leading to more accurate and nuanced predictions.
    """)
    
    
    

def st_dl2():
    st.image("./data/images/mlpipeline.png")
    st.markdown("""

                """)

def st_dl3():
    st.markdown("""
In this model, units represent the number of neurons in each LSTM layer. return_sequences=True is crucial in the first layers to ensure the output includes sequences, which are essential for stacking LSTM layers. The final LSTM layer does not return sequences as we prepare the data for the attention layer.                
   
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

