# Sentiment analysis using Bidirectional LSTM (Natural Language Processing)
While building this project our objective was to predict the sentiments of people (positive,1/negative,0) on different movies based on the given feature like Review and Sentiments. I have performed all the steps from data gathering to the model deployment. During Model evaluation I compared various ML models (Naive Bayes, Logistic Regression, Support Vector Machines etc.) on the basis of accuracy_score metric and find the best one, then I also created a neural network to check its performance against the ML models that I created, what I found is that ML models were not giving me that much val_accuracy (around 75-80%) in comparison to Bidirectional LSTM Neural Network (val_accuracy around 85-86%). So in this module I have predicted whether the people's sentiments are positive or not based on "IMDB Movie Reviews Dataset" using Bidirectional LSTM Neural Network.

This project can be deployed in flask but there are some problems with its deployment in heroku, so currenlty this project is in its deployment stage in heroku platform.
