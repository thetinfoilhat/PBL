import streamlit as st
#page config
st.set_page_config(page_title="How It Works", page_icon="🤔")

#begin actual stuff
st.markdown("# NLP Models 101")
st.markdown("Disclaimer: I am not a trained professional and I most likely have no idea what I'm doing")

#Introduction

st.markdown(f"### **Introduction**")
st.markdown(
    f"""
    A NLP model is a probabilistic statistical model that determines the probability of a given sequence of words occurring 
    in a sentence based on the previous words. NLP models also exist as language models that are developed using neural networks
    (which is what we are doing). NLP models work by finding relationships between the constituent parts of language 
    — for example, the model will find relationships between the letters, words, and sentences found in a text dataset that it is 
    given for training. In our case, the model takes in data containing articles, their titles, and a label indicating whether
    the given text is true or false. Using this, the model is able to accurately predict words and patterns that indicate
    inaccurate information.  
"""
)
st.markdown(f"### **Code**")
st.markdown(
    f"""
    To make this, I decided to use python as our language of choice at it contains useful libraries and is suited for
    sifting through large amounts of boring data.\nWe first import a bunch of libraries, their dependencies, and read our dataset with pandas. 
    To be honest I don't even know what libraries in this list are used and what aren't so I decided to err on the side of caution 
    and import more than needed. (From now on somewhat useless code is hidden under dropdowns, you can thank me later.)
"""
)
lib_code = """import numpy as np
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

import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
tf.disable_eager_execution()

# Reading the data
data = pd.read_csv("news.csv")
data.head()"""

with st.expander("Libraries that need to be imported"):
    st.code(lib_code, language="python")

st.markdown(
    f"""We then encode the data and make some interesting variables with very boring names."""
)

var_code = """le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])
embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1"""

with st.expander("Variables!"):
    st.code(var_code, language='python')

st.markdown("***Tokenization***")
st.markdown(
    f"""Now here's the cool stuff, we make something called a tokenizer that tokenizes (obviously) the neural network's input text.
     This process divides a large piece of continuous text into distinct units or tokens basically."""
)

st.markdown(
    f"""Here I just split the data by column"""
)

token_code = """title = []
text = []
labels = []
for x in range(training_size):
    title.append(data['title'][x])
    text.append(data['text'][x])
    labels.append(data['label'][x])"""

st.code(token_code, language='python')

st.markdown(
    """And then we just apply the tokenizer. Tensorflow already has this set up for us so it's a quick call to
    the tokenizer class and then the cool stuff is done."""
)

token1_code = """tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(title)
word_index1 = tokenizer1.word_index
vocab_size1 = len(word_index1)
sequences1 = tokenizer1.texts_to_sequences(title)
padded1 = pad_sequences(
    sequences1,  padding=padding_type, truncating=trunc_type)
split = int(test_portion * training_size)
training_sequences1 = padded1[split:training_size]
test_sequences1 = padded1[0:split]
test_labels = labels[0:split]
training_labels = labels[split:training_size]"""

st.code(token1_code, language='python')



st.markdown("**Word Embedding**")
st.markdown(
    """Here we just use a file called glove.whatever.txt that has a predefined vector space for words. It allows
    words with similar meaning to be represented equally."""
)
embed_code = """embeddings_index = {}
    with open('glove.6B.50d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Generating embeddings
    embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim))
    for word, i in word_index1.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    """
with st.expander(f"More for loops"):
    st.code(embed_code, language='python')

st.markdown("**Creating Model Architecture**")
st.markdown(
    """Now it's time to use some Tensorflow to give this model some life. We use the Tensorflow embedding technique
    with Keras Embedding Layer to map the input data into some set of dimensions. We're essentially
    creating neurons according to Google. We then train it at 50 epochs (generations) and let the code learn 
    by itself for 30 seconds."""
)
model_code = """
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim,
                              input_length=max_length, weights=[
                                  embeddings_matrix],
                              trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

#Defining training parameters
num_epochs = 50
  
training_padded = np.array(training_sequences1)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences1)
testing_labels = np.array(test_labels)
  
history = model.fit(training_padded, training_labels, 
                    epochs=num_epochs,
                    validation_data=(testing_padded,
                                     testing_labels), 
                    verbose=2)"""
st.code(model_code, language='python')

st.markdown("**Finishing Touches**")
st.markdown(
    """Our model is now done training and then we save the weights and biases it generated throughout its training into
    a .h5 file (I have no clue what that is). We also save the tokenizer so we can use it again without having to write out 20 lines
    of code."""
)
final_code = """model.save('model.h5')
tokenizer_json = tokenizer1.to_json()
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer1, f)"""

st.code(final_code, language='python')

st.markdown(
    """And just like that we're done! Most of the work here is taken care of by the libraries, all you need to do now is to
    load the model to whatever file or project you're working on. The full functionality of the model is contained
    in the .h5 and .json files that we exported in the previous lines. Hopefully this mini-explanation was helpful, 
    if you have any questions you can come along with me and Google our way to enlightment."""
)
st.write("\n")
st.markdown("Thanks for reading! 😅")


