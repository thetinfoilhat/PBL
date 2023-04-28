import streamlit as st

st.set_page_config(
    page_title="Home Page",
    page_icon="👋",
)

st.write("# Welcome to our app! 👋")


st.markdown(
    f"""
    This app is made to help combat minformation encountered in daily life.
    The project utilizes the Tensorflow library and keras to create a neural network that can detect fake news with almost 97% accuracy. 
    Some cool stuff like word embedding, tokenizers, and lots of math were needed to make this work.\n 
    **👈 Select the news classifier page** to see what our AI can do!
    ### Want to learn more?
    - Check out [keras.io](https://keras.io/)
    - Take a look at all of our [code](https://github.com)
    - Explore our amazing app!
"""
)