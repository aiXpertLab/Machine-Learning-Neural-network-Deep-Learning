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


def dl_general():
    st.image("./data/images/mlpipeline.png")
    st.markdown(general)    
    st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")


def dl_theory():
    st.header("🧠1. Long Short-Term Memory networks")
    st.markdown("""

        LSTM networks are a type of Recurrent Neural Network (RNN) specially designed to remember and process sequences of data over long periods. 
        What sets LSTMs apart from traditional RNNs is their ability to preserve information for long durations, courtesy of their unique structure comprising three gates: the input, forget, and output gates.
        These gates collaboratively manage the flow of information, deciding what to retain and what to discard, thereby mitigating the issue of vanishing gradients — a common problem in standard RNNs.
    
    """)
    st.image("./data/images/lstm.png")
    
    st.header("👩‍🏫2. Attention Mechanism")
    st.markdown("""
        The attention mechanism, initially popularized in the field of natural language processing, has found its way into various other domains, including finance. 
        It operates on a simple yet profound concept: not all parts of the input sequence are equally important. 
        By allowing the model to focus on specific parts of the input sequence while ignoring others, the attention mechanism enhances the model’s context understanding capabilities.

        Incorporating attention into LSTM networks results in a more focused and context-aware model. 
        When predicting stock prices, certain historical data points may be more relevant than others. 
        The attention mechanism empowers the LSTM to weigh these points more heavily, leading to more accurate and nuanced predictions.

        tensorflow两种attention机制，分别为Bahdanau attention，和LuongAttention.
        Attention 解决了 RNN 不能并行计算的问题。Attention机制每一步计算不依赖于上一步的计算结果，因此可以和CNN一样并行处理。
        模型复杂度跟 CNN、RNN 相比，复杂度更小，参数也更少。所以对算力的要求也就更小。
        在 Attention 机制引入之前，有一个问题大家一直很苦恼：长距离的信息会被弱化，就好像记忆能力弱的人，记不住过去的事情是一样的。

        Attention 是挑重点，就算文本比较长，也能从中间抓住重点，不丢失重要的信息。下图红色的预期就是被挑出来的重点。

        Attention 经常会和 Encoder–Decoder 一起说，之前的文章《一文看懂 NLP 里的模型框架 Encoder-Decoder 和 Seq2Seq》 也提到了 Attention。
    """)
    st.image("./data/images/attention.gif")
    st.header("Attention 原理的3步分解：")
    st.image("./data/images/attentionpipeline.png")
    st.markdown("""

        第一步： query 和 key 进行相似度计算，得到权值

        第二步：将权值进行归一化，得到直接可用的权重

        第三步：将权重和 value 进行加权求和

        从上面的建模，我们可以大致感受到 Attention 的思路简单，四个字“带权求和”就可以高度概括，大道至简。做个不太恰当的类比，人类学习一门新语言基本经历四个阶段：死记硬背（通过阅读背诵学习语法练习语感）->提纲挈领（简单对话靠听懂句子中的关键词汇准确理解核心意思）->融会贯通（复杂对话懂得上下文指代、语言背后的联系，具备了举一反三的学习能力）->登峰造极（沉浸地大量练习）。

        这也如同attention的发展脉络，RNN 时代是死记硬背的时期，attention 的模型学会了提纲挈领，进化到 transformer，融汇贯通，具备优秀的表达学习能力，再到 GPT、BERT，通过多任务大规模学习积累实战经验，战斗力爆棚。

        要回答为什么 attention 这么优秀？是因为它让模型开窍了，懂得了提纲挈领，学会了融会贯通。

        **Attention 的 N 种类型**
        Attention 有很多种不同的类型：Soft Attention、Hard Attention、静态Attention、动态Attention、Self Attention 等等。下面就跟大家解释一下这些不同的 Attention 都有哪些差别。

        1. 计算区域

        根据Attention的计算区域，可以分成以下几种：

        1）Soft Attention，这是比较常见的Attention方式，对所有key求权重概率，每个key都有一个对应的权重，是一种全局的计算方式（也可以叫Global Attention）。这种方式比较理性，参考了所有key的内容，再进行加权。但是计算量可能会比较大一些。

        2）Hard Attention，这种方式是直接精准定位到某个key，其余key就都不管了，相当于这个key的概率是1，其余key的概率全部是0。因此这种对齐方式要求很高，要求一步到位，如果没有正确对齐，会带来很大的影响。另一方面，因为不可导，一般需要用强化学习的方法进行训练。（或者使用gumbel softmax之类的）

        3）Local Attention，这种方式其实是以上两种方式的一个折中，对一个窗口区域进行计算。先用Hard方式定位到某个地方，以这个点为中心可以得到一个窗口区域，在这个小区域内用Soft方式来算Attention。

        2. 所用信息

        假设我们要对一段原文计算Attention，这里原文指的是我们要做attention的文本，那么所用信息包括内部信息和外部信息，内部信息指的是原文本身的信息，而外部信息指的是除原文以外的额外信息。

        1）General Attention，这种方式利用到了外部信息，常用于需要构建两段文本关系的任务，query一般包含了额外信息，根据外部query对原文进行对齐。

        比如在阅读理解任务中，需要构建问题和文章的关联，假设现在baseline是，对问题计算出一个问题向量q，把这个q和所有的文章词向量拼接起来，输入到LSTM中进行建模。那么在这个模型中，文章所有词向量共享同一个问题向量，现在我们想让文章每一步的词向量都有一个不同的问题向量，也就是，在每一步使用文章在该步下的词向量对问题来算attention，这里问题属于原文，文章词向量就属于外部信息。

        2）Local Attention，这种方式只使用内部信息，key和value以及query只和输入原文有关，在self attention中，key=value=query。既然没有外部信息，那么在原文中的每个词可以跟该句子中的所有词进行Attention计算，相当于寻找原文内部的关系。

        还是举阅读理解任务的例子，上面的baseline中提到，对问题计算出一个向量q，那么这里也可以用上attention，只用问题自身的信息去做attention，而不引入文章信息。

        3. 结构层次

        结构方面根据是否划分层次关系，分为单层attention，多层attention和多头attention：

        1）单层Attention，这是比较普遍的做法，用一个query对一段原文进行一次attention。

        2）多层Attention，一般用于文本具有层次关系的模型，假设我们把一个document划分成多个句子，在第一层，我们分别对每个句子使用attention计算出一个句向量（也就是单层attention）；在第二层，我们对所有句向量再做attention计算出一个文档向量（也是一个单层attention），最后再用这个文档向量去做任务。

        3）多头Attention，这是Attention is All You Need中提到的multi-head attention，用到了多个query对一段原文进行了多次attention，每个query都关注到原文的不同部分，相当于重复做多次单层attention：


        最后再把这些结果拼接起来：


        4. 模型方面

        从模型上看，Attention一般用在CNN和LSTM上，也可以直接进行纯Attention计算。

        1）CNN+Attention

        CNN的卷积操作可以提取重要特征，我觉得这也算是Attention的思想，但是CNN的卷积感受视野是局部的，需要通过叠加多层卷积区去扩大视野。另外，Max Pooling直接提取数值最大的特征，也像是hard attention的思想，直接选中某个特征。

        CNN上加Attention可以加在这几方面：

        a. 在卷积操作前做attention，比如Attention-Based BCNN-1，这个任务是文本蕴含任务需要处理两段文本，同时对两段输入的序列向量进行attention，计算出特征向量，再拼接到原始向量中，作为卷积层的输入。

        b. 在卷积操作后做attention，比如Attention-Based BCNN-2，对两段文本的卷积层的输出做attention，作为pooling层的输入。

        c. 在pooling层做attention，代替max pooling。比如Attention pooling，首先我们用LSTM学到一个比较好的句向量，作为query，然后用CNN先学习到一个特征矩阵作为key，再用query对key产生权重，进行attention，得到最后的句向量。

        2）LSTM+Attention

        LSTM内部有Gate机制，其中input gate选择哪些当前信息进行输入，forget gate选择遗忘哪些过去信息，我觉得这算是一定程度的Attention了，而且号称可以解决长期依赖问题，实际上LSTM需要一步一步去捕捉序列信息，在长文本上的表现是会随着step增加而慢慢衰减，难以保留全部的有用信息。

        LSTM通常需要得到一个向量，再去做任务，常用方式有：

        a. 直接使用最后的hidden state（可能会损失一定的前文信息，难以表达全文）

        b. 对所有step下的hidden state进行等权平均（对所有step一视同仁）。

        c. Attention机制，对所有step的hidden state进行加权，把注意力集中到整段文本中比较重要的hidden state信息。性能比前面两种要好一点，而方便可视化观察哪些step是重要的，但是要小心过拟合，而且也增加了计算量。

        3）纯Attention

        Attention is all you need，没有用到CNN/RNN，乍一听也是一股清流了，但是仔细一看，本质上还是一堆向量去计算attention。
    """)
    st.image("./data/images/attentiontypes.png")




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

