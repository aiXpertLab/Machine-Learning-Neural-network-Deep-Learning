import streamlit as st
from utils import st_def, tab_dl
st_def.st_logo(title = "2006 👋 Deep Learning!", page_title="2006 Deep Learning",)

tab1, tab2, tab3, t4 = st.tabs(["General", "Theory", "Vision", 'CNN'])

with tab1:  tab_dl.dl_general()
with tab2:  tab_dl.dl_theory()
with tab3:  
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
    
    
with t4: 
    st.write('''
    CNN本质上是一个多层感知机，其成功的原因关键在于它所采用的局部连接和共享权值的方式，一方面减少了的权值的数量使得网络易于优化，另一方面降低了模型复杂度，降低了过拟合的风险。CNN是一个前溃式神经网络，能从一个二维图像中提取其拓扑结构，采用反向传播算法来优化网络结构，求解网络中的未知参数。CNN具有一些传统技术所没有的优点：良好的容错能力、并行处理能力和自学习能力，可处理环境信息复杂，背景知识不清楚，推理规则不明确情况下的问题，允许样品有较大的缺损、畸变，运行速度快，自适应性能好，具有较高的分辨率。它是通过结构重组和减少权值将特征抽取功能融合进多层感知器，省略识别前复杂的图像特征抽取过程。
    
    CNN网络一共有5个层级结构：
    
        - 输入层
        - 卷积层
        - 激活层
        - 池化层
        - 全连接FC层
     ''')