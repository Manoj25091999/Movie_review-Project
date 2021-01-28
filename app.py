# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:18:38 2021

@author: Manoj Kumar
"""

import pandas as pd
from flask import Flask, render_template, url_for, request
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np


# load the model from disk
filename = 'nlp1_model.h5'
model = load_model(filename)
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        # Converting the input sentence into a one_hot feature vector
        one_hot_repr = [one_hot(sent,10000) for sent in data]
        # Finding the length of the string we input and then embedding it
        embedded_repr = pad_sequences(one_hot_repr, padding='pre', maxlen=9168)
        x_test = np.array(embedded_repr)
        my_prediction = model.predict(x_test)
        my_prediction = (my_prediction>0.5)
        return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)