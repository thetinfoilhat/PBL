# Import a bunch of libraries I probably don't need
import streamlit as st
#page config
st.set_page_config(page_title="News Classifier", page_icon="🔎")
import numpy as np
import pandas as pd
import json
import csv
import random
import tensorflow
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from keras.models import load_model

import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
tf.disable_eager_execution()


#Define tokenizer variables
trunc_type = 'post'
padding_type = 'post'


# Load tokenizer and model
with open('pages/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('pages/model.h5')


# Set up streamlit app
st.title("News Classifier")


# Get user input
with st.form("my_form", clear_on_submit= True):
    # Render the text area with a placeholder
    user_input = st.text_area("", value="", key="headline", max_chars=1000, placeholder="Enter some news...")
    # Render the submit button
    submitted = st.form_submit_button("Predict")

#Check for user input and a true submission value
if user_input and submitted:
    # Perform classification on user input
    sequences = tokenizer.texts_to_sequences([user_input])[0]
    sequences = pad_sequences([sequences], maxlen=54, padding='post', truncating='post')
    prediction = model.predict(sequences, verbose=0)[0][0]
    
    # Show prediction and percentage
    st.write('\n')
    st.markdown("## Prediction:")
    if prediction >= 0.5:
        st.markdown("### **REAL**")
    else:
        st.markdown("### **FAKE**")
    st.write("\n")
    progress_prediction = int(prediction * 100)
    percentage = round(prediction*100, 2)
    progress_bar = st.progress(progress_prediction)
    st.write(f"This news is {percentage}% true.")

    #button code that does nothing for now...
    st.write("Give us some feedback if we're wrong!")

    # Wrap the buttons inside a container with flexbox style
    
    col1, col2 = st.columns([1,1], gap='small')
    with col1:
        if st.button('Correct', key='real', use_container_width = True):
            st.write("Thanks for the feedback!")

    with col2:
        if st.button('Incorrect', key='fake', use_container_width = True):
            st.write("Thanks for the feedback!")


