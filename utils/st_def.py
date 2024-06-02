import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

def st_sidebar():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.write("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
        add_vertical_space(2)
        st.write('Made with ❤️ by [aiXpertLab](https://hypech.com)')

    return openai_api_key

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
