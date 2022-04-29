# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 06:59:26 2022

@author: rayan
"""

# imports
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text

# import data set
df = pd.read_csv("Youtube04-Eminem.csv")

# check the number of unique spams and non-spam comments in the data set
print((df.CLASS==1).sum()) # spam
print((df.CLASS==0).sum()) # not-spam

# remove columns that are useless
df = df.drop(columns=["COMMENT_ID", "AUTHOR", "DATE"])

#remove stop words
stop = set(stopwords.words("english"))

# function to remove stop words
def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)

# removing stop words on the CONTENT column in the data frame
df['CONTENT'] = df.CONTENT.map(remove_stopwords)

# split into testing and training data
X_train, X_test, y_train, y_test = train_test_split(df['CONTENT'], df['CLASS'], test_size=0.3, random_state=22)

# download BERT models from kersLayer hub
bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# function to encode the sentences in the dataframe and output them as a pooled_output 
def encodeSentences(sent):
    pre_processed_text = bert_preprocess(sent)
    return bert_encoder(pre_processed_text)['pooled_output']

# using the function on the data frame
df = encodeSentences(df.CONTENT)

# BERT layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="CONTENT")
pre_processed_text = bert_preprocess(text_input)
outputs = bert_encoder(pre_processed_text)

# Neural network layers
layer = tf.keras.layers.Dropout(0.1, name="droput")(outputs['pooled_output'])

layer = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(layer)

# create the final model
model = tf.keras.Model(inputs=[text_input], outputs=[layer])

# metrics 
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall")
    ]

# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = METRICS)

model.fit(X_train, y_train, epochs=20)

model.evaluate(X_test, y_test)

model.save("youtubeSpamModel.h5")

predict = model.predict(X_test)





