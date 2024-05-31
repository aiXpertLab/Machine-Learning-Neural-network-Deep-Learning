import streamlit as st

def dl_general():
    st.image('./images/dl/main.png')
    general="""
            1. Loading Dataset
            2. Chose Model (including activation/relu/softmax): `model = keras.Sequential([layers.Dense(512, activation="relu"),layers.Dense(10, activation="softmax")])`
            3. Compile (optimizer, loss function, metric)
            4. Reshape preprocessing data.
            5. 
            
            
            
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
    st.markdown(general)    
    st.info("The Exchange Of Methods And Algorithms Between Human And Machine To Deep Learn And Apply Problem Solving Is Known As Deep Learning (DL) ― P.S. Jagadeesh Kumar")


def dl_theory():
    st.header("🧠1. Long Short-Term Memory networks")
    st.markdown("""

        LSTM networks are a type of Recurrent Neural Network (RNN) specially designed to remember and process sequences of data over long periods. 
        What sets LSTMs apart from traditional RNNs is their ability to preserve information for long durations, courtesy of their unique structure comprising three gates: the input, forget, and output gates.
        These gates collaboratively manage the flow of information, deciding what to retain and what to discard, thereby mitigating the issue of vanishing gradients — a common problem in standard RNNs.
    
    """)
    st.image("./images/lstm.png")
    
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
    st.image("./images/attention.gif")
    st.header("Attention 原理的3步分解：")
    st.image("./images/attentionpipeline.png")
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
    st.image("./images/attentiontypes.png")


def dl_1():
    st.image("./images/mlpipeline.png")
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
    
    
    

def dl_vision():
    st.image("./images/zhang3.gif")
    st.write('''One method from deep learning that deserves the most attention for application in computer vision is: `Convolutional Neural Networks (CNNs)`.
             Additionally, both of the following network types may be useful for interpreting or developing
inference models from the features learned and extracted by CNNs; they are:
- Multilayer Perceptrons (MLP).
- Recurrent Neural Networks (RNNs).

The MLP or fully-connected type neural network layers are useful for developing models that make predictions given the learned features extracted by CNNs. RNNs, such as LSTMs,
may be helpful when working with sequences of images over time, such as with video.

Deep learning will not solve computer vision or artificial intelligence. To date, deep learning
methods have been evaluated on a broader suite of problems from computer vision and achieved
success on a small set, where success suggests performance or capability at or above what was
previously possible with other methods. Importantly, those areas where deep learning methods
are showing the greatest success are some of the more end-user facing, challenging, and perhaps
more interesting problems. Five examples include:
- Optical Character Recognition.
- Image Classification.
- Object Detection.
- Face Detection.
- Face Recognition.

All five tasks are related under the umbrella of `object recognition`, which refers to tasks that involve identifying, localizing, and/or extracting specific content from digital photographs

             ''')
    

def dl_cnn():
    st.image('./images/dl/t4.png')
    st.write('''

一、从前馈神经网络说起

1.必会的内功：前馈神经网络

前馈神经网络（Feedforward Neural Networks）是最基础的神经网络模型，也被称为多层感知机（MLP）。它由多个神经元组成，每个神经元与前一层的所有神经元相连，形成一个“全连接”的结构。每个神经元会对其输入数据进行线性变换（通过权重矩阵），然后通过一个非线性函数（如ReLU或Sigmoid）进行激活。这就是前馈神经网络的基本操作。
CNN就是一种特殊的前馈神经网络。这两者的主要区别在于，CNN在前馈神经网络的基础上加入了卷积层和池化层（下边会讲到），以便更好地处理图像等具有空间结构的数据。

CNN就是在此基础上，将全连接层换成卷积层，并在ReLU层之后加入池化层（非必须），那么一个基本的CNN结构就可以表示成这样：

CNN本质上是一个多层感知机，其成功的原因关键在于它所采用的局部连接和共享权值的方式，一方面减少了的权值的数量使得网络易于优化，另一方面降低了模型复杂度，降低了过拟合的风险。CNN是一个前溃式神经网络，能从一个二维图像中提取其拓扑结构，采用反向传播算法来优化网络结构，求解网络中的未知参数。CNN具有一些传统技术所没有的优点：良好的容错能力、并行处理能力和自学习能力，可处理环境信息复杂，背景知识不清楚，推理规则不明确情况下的问题，允许样品有较大的缺损、畸变，运行速度快，自适应性能好，具有较高的分辨率。它是通过结构重组和减少权值将特征抽取功能融合进多层感知器，省略识别前复杂的图像特征抽取过程。

CNN网络一共有5个层级结构：

- 输入层
- 卷积层
- 激活层
- 池化层
- 全连接FC层
     ''')
    
def dl_mlp():
    # https://zhuanlan.zhihu.com/p/65472471
    st.image('./images/dl/t1.png')
    st.markdown("""
            #### 任务描述        
            我们已知四个数据点(1,1)(-1,1)(-1,-1)(1,-1)，这四个点分别对应I~IV象限，如果这时候给我们一个新的坐标点（比如(2,2)），那么它应该属于哪个象限呢？（没错，当然是第I象限，但我们的任务是要让机器知道）

            “分类”是神经网络的一大应用，我们使用神经网络完成这个分类任务。        """)
    st.image('./images/dl/t2.png')
    st.markdown("""
                这里我们构建一个两层神经网络，理论上两层神经网络已经可以拟合任意函数。
                1.1.输入层

                在我们的例子中，输入层是坐标值，例如（1,1），这是一个包含两个元素的数组，也可以看作是一个1*2的矩阵。输入层的元素维度与输入量的特征息息相关，如果输入的是一张32*32像素的灰度图像，那么输入层的维度就是32*32。

                1.2.从输入层到隐藏层

                连接输入层和隐藏层的是W1和b1。由X计算得到H十分简单，就是矩阵运算：


                如果你学过线性代数，对这个式子一定不陌生。如上图中所示，在设定隐藏层为50维（也可以理解成50个神经元）之后，矩阵H的大小为（1*50）的矩阵。

                1.3.从隐藏层到输出层

                连接隐藏层和输出层的是W2和b2。同样是通过矩阵运算进行的：                
                        
                1.4.分析

                通过上述两个线性方程的计算，我们就能得到最终的输出Y了，但是如果你还对线性代数的计算有印象的话，应该会知道：一系列线性方程的运算最终都可以用一个线性方程表示。也就是说，上述两个式子联立后可以用一个线性方程表达。对于两次神经网络是这样，就算网络深度加到100层，也依然是这样。这样的话神经网络就失去了意义。

                所以这里要对网络注入灵魂：激活层。

                ##### 2.激活层
                简而言之，激活层是为矩阵运算的结果添加非线性的。常用的激活函数有三种，分别是阶跃函数、Sigmoid和ReLU。不要被奇怪的函数名吓到，其实它们的形式都很简单，如下图：

                其中，阶跃函数输出值是跳变的，且只有二值，较少使用；Sigmoid函数在当x的绝对值较大时，曲线的斜率变化很小（梯度消失），并且计算较复杂；ReLU是当前较为常用的激活函数。

                需要注意的是，每个隐藏层计算（矩阵线性运算）之后，都需要加一层激活层，要不然该层线性计算是没有意义的。""")
    st.image('./images/dl/t3.png')
    st.markdown("""
                3.输出的正规化
                在图4中，输出Y的值可能会是(3,1,0.1,0.5)这样的矩阵，诚然我们可以找到里边的最大值“3”，从而找到对应的分类为I，但是这并不直观。我们想让最终的输出为概率，也就是说可以生成像(90%,5%,2%,3%)这样的结果，这样做不仅可以找到最大概率的分类，而且可以知道各个分类计算的概率值。

                具体是怎么计算的呢？

                计算公式如下：


                简单来说分三步进行：（1）以e为底对所有元素求指数幂；（2）将所有指数幂求和；（3）分别将这些指数幂与该和做商。

                这样求出的结果中，所有元素的和一定为1，而每个元素可以代表概率值。

                我们将使用这个计算公式做输出结果正规化处理的层叫做“Softmax”层。此时的神经网络将变成如下图所示：


                图5.输出正规化之后的神经网络
                4.如何衡量输出的好坏
                通过Softmax层之后，我们得到了I，II，III和IV这四个类别分别对应的概率，但是要注意，这是神经网络计算得到的概率值结果，而非真实的情况。

                比如，Softmax输出的结果是(90%,5%,3%,2%)，真实的结果是(100%,0,0,0)。虽然输出的结果可以正确分类，但是与真实结果之间是有差距的，一个优秀的网络对结果的预测要无限接近于100%，为此，我们需要将Softmax输出结果的好坏程度做一个“量化”。

                一种直观的解决方法，是用1减去Softmax输出的概率，比如1-90%=0.1。不过更为常用且巧妙的方法是，求对数的负数。

                还是用90%举例，对数的负数就是：-log0.9=0.046

                可以想见，概率越接近100%，该计算结果值越接近于0，说明结果越准确，该输出叫做“交叉熵损失（Cross Entropy Error）”。

                我们训练神经网络的目的，就是尽可能地减少这个“交叉熵损失”。

                此时的网络如下图：


                图6.计算交叉熵损失后的神经网络
                5.反向传播与参数优化
                上边的1~4节，讲述了神经网络的正向传播过程。一句话复习一下：神经网络的传播都是形如Y=WX+b的矩阵运算；为了给矩阵运算加入非线性，需要在隐藏层中加入激活层；输出层结果需要经过Softmax层处理为概率值，并通过交叉熵损失来量化当前网络的优劣。

                算出交叉熵损失后，就要开始反向传播了。其实反向传播就是一个参数优化的过程，优化对象就是网络中的所有W和b（因为其他所有参数都是确定的）。

                神经网络的神奇之处，就在于它可以自动做W和b的优化，在深度学习中，参数的数量有时会上亿，不过其优化的原理和我们这个两层神经网络是一样的。

                这里举一个形象的例子描述一下这个参数优化的原理和过程：

                假设我们操纵着一个球型机器行走在沙漠中


                我们在机器中操纵着四个旋钮，分别叫做W1，b1，W2，b2。当我们旋转其中的某个旋钮时，球形机器会发生移动，但是旋转旋钮大小和机器运动方向之间的对应关系是不知道的。而我们的目的就是走到沙漠的最低点。


                此时我们该怎么办？只能挨个试喽。

                如果增大W1后，球向上走了，那就减小W1。

                如果增大b1后，球向下走了，那就继续增大b1。

                如果增大W2后，球向下走了一大截，那就多增大些W2。

                。。。

                这就是进行参数优化的形象解释（有没有想到求导？），这个方法叫做梯度下降法。

                当我们的球形机器走到最低点时，也就代表着我们的交叉熵损失达到最小（接近于0）。

                关于反向传播，还有许多可以讲的，但是因为内容较多，就放在下一篇文章中说吧。不过上述例子对于理解神经网络参数优化的过程，还是很有帮助的。

                6.迭代

                神经网络需要反复迭代。

                如上述例子中，第一次计算得到的概率是90%，交叉熵损失值是0.046；将该损失值反向传播，使W1,b1,W2,b2做相应微调；再做第二次运算，此时的概率可能就会提高到92%，相应地，损失值也会下降，然后再反向传播损失值，微调参数W1,b1,W2,b2。依次类推，损失值越来越小，直到我们满意为止。

                此时我们就得到了理想的W1,b1,W2,b2。

                此时如果将任意一组坐标作为输入，利用图4或图5的流程，就能得到分类结果。
                """)



def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)
def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_0():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_4():
    st.markdown("""
The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:
                """)

def dl_6():
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

def dl_11():
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

