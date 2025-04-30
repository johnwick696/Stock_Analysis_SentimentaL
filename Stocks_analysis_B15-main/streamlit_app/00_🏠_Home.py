import streamlit as st

st.set_page_config(
    page_title="Stock Analysis App",
    page_icon="😎",
)

with open("designing.css") as source_des:
    st.markdown(f'<style>{source_des.read()}</style>', unsafe_allow_html=True)



st.markdown("""
    <style>
    body {
        color: #fff;
        background-color: #4F4F4F;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("# **🏠 Stock Trading Prediction**")
st.markdown("""
We are a team of passionate individuals who believe in the power of data. 
Our mission is to provide easy and accessible tools for everyone to understand, analyze, and predict the stock market. 
With our application, we aim to enhance your stock market insights by providing real-time data, analysis, and prediction capabilities.
""")
st.markdown("## **Our Team**")
st.markdown("""
- Nimit Agarwal (E22CSEU0423)\n
- Shivansh Panday (E22CSEU1688)\n
- Prabhat Thakur (E22CSEU0448)
""")
st.markdown("## **Contributions**")
st.markdown("""
This application is a result of the collective efforts of our team. 
Nimit Agarwal , our lead developer has played a pivotal role in bringing this project to life. 
His contributions have been instrumental in the development and success of this application, particularly in the areas of data analysis and stock prediction.
""")
st.markdown("## **Contact Us**")
st.markdown("""
Email :- nagarwal2526@gmail.com
""")

