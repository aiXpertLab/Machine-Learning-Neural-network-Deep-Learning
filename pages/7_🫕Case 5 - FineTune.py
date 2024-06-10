import streamlit as st
from streamlit_extras.stateful_button import button
from utils import st_def

st_def.st_logo(title = "👋Dogs and Cats", page_title="Summary",)

t1, t2, t3, t4, t5, t6 = st.tabs(["Llama3", " ", " ", '', '',''])

with t1: 
    # https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations
    st.write('The way everyone should be thinking about fine-tuning, is not how we can outperform OpenAI or replace RAG, but how we can maintain the same performance while cutting down on inference time and cost for your specific use case.')
    st.write('''Fine-tuning comes in two different forms:
             
            - SFT (Supervised Fine-Tuning): LLMs are fine-tuned on a set of instructions and responses. The model’s weights will be updated to minimize the difference between the generated output and labeled responses.
            - RLHF (Reinforcement Learning from Human Feedback): LLMs are trained to maximize the reward function (using Proximal Policy Optimization Algorithms or the Direct Preference Optimization (DPO) algorithm). This technique uses feedback from human evaluation of generated outputs, which in turn captures more intricate human preferences, but is prone to inconsistent human feedback.
             
             ''')
    st.subheader('Step 1—Installation')
    st.write("""
        - transformers: to load models, tokenizers, etc.
        - peft: to perform parameter efficient fine-tuning
        - bitsandbytes: to setup 4-bit quantization
        - trl: for supervised fine-tuning
        - deepeval: to evaluate the fine-tuned LLM
    """)
    
    st.subheader('Step 2— Quantization Setup')
    st.write('''
             To optimize Colab RAM usage during LLaMA-3 8B fine-tuning, we use QLoRA (quantized low-rank approximation). Here’s a breakdown of its key principles:

- 4-Bit Quantization: QLoRA compresses the pre-trained LLaMA-3 8B model by representing weights with only 4 bits (as opposed to standard 32-bit floating-point). This significantly shrinks the model’s memory footprint.
- Frozen Pre-trained Model: After quantization, the vast majority of LLaMA-3’s parameters are frozen. This prevents direct updates to the core model during fine-tuning.
- Low-Rank Adapters: QLoRA introduces lightweight, trainable adapter layers into the model’s architecture. These adapters capture task-specific knowledge without drastically increasing the number of parameters.
- Gradient-Based Fine-tuning: During the fine-tuning process, gradients flow through the frozen 4-bit quantized model but are used to update solely the parameters within the low-rank adapters. This isolated optimization greatly reduces computational overhead.
             ''')
    
    st.subheader('Step 3 — Load LLaMA-3 with QLoRA Configuration')
    st.subheader('Step 4 — Load Tokenizer')
    st.write('''
             
When an LLM reads text, it first has to convert the text to a readable format. This process is known a tokenization, which is carried out by a tokenizer.

Tokenizers are usually designed to work with their respective models.
             
                 ''')



    st.write('''Step 5 — Load Dataset
As explained in the previous section, we’ll be using the mlabonne/guanaco-llama2–1k dataset for fine-tuning given its high quality data labels and compatibility with LLaMA-3's prompt templates.
                 ''')



    st.write('''
Step 6 — Load LoRA Configurations for PEFT
I’m not going to go into the detailed differences between QLoRA and LoRA, but LoRA is essentially a less memory-efficient version of QLoRA since it does not use quantization, but may yield slightly higher accuracy. (You can read more about LoRA here.)

In this step, we configure LoRA for parameter efficient fine-tuning (PEFT), which updates a small subset of parameters in contrast to normal fine-tuning, where all model parameters are updated instead.

             
             
                 ''')



    st.write('''
Step 7 — Set Training Arguments and SFT Parameters
We’re almost there, all that’s left is to set the arguments required for training and the supervised fine-tuning (SFT) parameters for the trainer:             
             
                 ''')



    st.write('''
             
             
                 ''')



    st.write('''
             
             
                 ''')



    st.write('''
             
             
                 ''')



    st.write('''
             
             
                 ''')
