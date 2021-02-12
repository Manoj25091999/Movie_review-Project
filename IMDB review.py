# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GT-_fMxp-hpDeSCCNENcOXOyOAgCBnWk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

data = pd.read_csv('/content/drive/MyDrive/IMDB Project/IMDB Dataset.csv')

data.head()

data.tail()

# Checking null values
data.isnull().sum()

data.info()

"""Dividing the datasets for analysis and modelling"""

Y = data['sentiment']

X = data.drop(['sentiment'], axis=1)

X.shape, Y.shape

Y.value_counts(normalize=True)

Y.replace({'positive':1, 'negative':0}, inplace=True)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

pip install nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

ps = PorterStemmer()
lt = WordNetLemmatizer()

# Removing the stopwords
corpus = []
for i in range(0,len(X)):
  review = re.sub("[^a-zA-Z]", " ", X['review'][i])
  review = review.lower()
  review = review.split()
  review = [lt.lemmatize(word) for word in review if not word in stopwords.words('english')]
  review = " ".join(review)
  corpus.append(review)

X['review'][1]

corpus[1]



"""## Using Bidirectional RNN with LSTM using word embeddings"""

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Converting sentences in corpus into a one_hot feature vector
one_hot_repr = [one_hot(sent,10000) for sent in corpus]
print(one_hot_repr[0])

with open('one_hot_transform.pkl', 'wb') as f:
  pickle.dump(one_hot_repr,f)

# Making each sentence into same length
sent_length = len(max(corpus, key=len)) #Finding the max length of a string in corpus
embedded_repr = pad_sequences(one_hot_repr, padding='pre', maxlen=sent_length)
print(embedded_repr)

embedded_repr[0]

pickle.dump(embedded_repr, open('embedded_repr.pkl','wb'))

"""## Modelling"""

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = Sequential()
# Adding Word Embedding layer
model.add(Embedding(10000,10, input_length=sent_length))
# Adding Bidirectional LSTM layer
model.add(layers.Bidirectional(layers.LSTM(100)))
# Adding output layer
model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
# Comipiling the model (adding optimizer, loss function, and required metrics)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

X_final = np.array(embedded_repr)
Y_final = np.array(Y)

X_final.shape, Y_final.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.33)

# Fitting the model
model_history = model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=128, epochs=10)

# Saving the model
model.save('nlp1_model.h5')

from tensorflow.keras.models import load_model

model_3 = load_model('nlp1_model.h5')

import numpy as np

sent = ['The movie is good']

one_hot_repr_2 = [one_hot(sent,10000) for sent in sent]

sent_length_2 = len(max(sent, key=len)) #Finding the max length of a string in corpus
embedded_repr_2 = pad_sequences(one_hot_repr_2, padding='pre', maxlen=9168)

sent_final = np.array(embedded_repr_2)

pred = model_3.predict(sent_final)

pred = (pred>0.5)

pred