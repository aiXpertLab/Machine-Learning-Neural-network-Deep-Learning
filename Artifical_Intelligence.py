# from hypecheth
import streamlit as st
from utils.st_def import st_main_contents, st_logo

st_logo(title='👋 Artificial Intelligence! 🍨 ', page_title="AI🍨",)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General", "Learning Material", "Developing Environment", "",""])

with tab1: st_main_contents()
with tab2:
    st.markdown('[FastAI Deep Learning](https://course.fast.ai/)',unsafe_allow_html=True)
with tab3:
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
    
    
