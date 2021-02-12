import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

data = pd.read_csv('IMDB Dataset.csv')

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

## Using Bidirectional RNN with LSTM using word embeddings

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

# Some extra things which I tried as an experiment

"""# Vectorzing the words
Vect = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = Vect.fit_transform(corpus).toarray()

X.shape

# Spliiting the dataset for modelling

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

Vect.get_feature_names()[:50]

Vect.get_params()

Checking the training dataset

final_df = pd.DataFrame(x_train, columns = Vect.get_feature_names())

final_df.head()

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

classifier_1 = SVC() 
classifier_2 = MultinomialNB()
classifier_3 = LogisticRegression()

classifier_2.fit(x_train, y_train)

x_test.shape

pred2 = classifier_2.predict(x_test)

pred2

pred_prob = classifier_2.predict_proba(x_test)
"""

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score

"""print("Accuracy score with Multinomial Naive Bayes : {:.4f}".format(accuracy_score(y_test,pred2)))

print("The classification_report:")
print(classification_report(y_test,pred2))

precision_score(y_test, pred2)

confusion_matrix(y_test, pred2)

classifier_3.fit(x_train,y_train)

pred3 = classifier_3.predict(x_test)

accuracy_score(y_test,pred3)

pred_prob = classifier_3.predict_proba(x_test)

pred_prob

Checking the metrics

# Function to calculate Precision and Recall

def calc_precision_recall(y_true, y_pred):
    
    # Convert predictions to series with index matching y_true
    y_pred = pd.Series(y_pred, index=y_true.index)
    
    # Instantiate counters
    TP = 0
    FP = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in y_true.index: 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_pred[i]==0 and y_test[i]!=y_pred[i]:
           FN += 1
    
    # Calculate true positive rate and false positive rate
    # Use try-except statements to avoid problem of dividing by 0
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1
    
    try:
        recall = TP / (TP + FN)
    except:
        recall = 1

    return precision, recall
"""



"""# Checking the thresholds for better prediction

lr_proba = pred_prob[:,1]

# Defining probability thresholds to use between 0 and 1
#prob_thres = np.linspace(0,1,num=100)

x_test_pred=[]
  
for l in lr_proba:
  if l>0.50:
   x_test_pred.append(1)
  if l<0.50:
   x_test_pred.append(0)

accuracy_score(y_test,x_test_pred)

print(classification_report(y_test,x_test_pred))

confusion_matrix(y_test, x_test_pred, labels=[0,1])

from sklearn.metrics import plot_precision_recall_curve

plot_precision_recall_curve(classifier_3, x_test, y_test, name = 'Logistic Regression');

# Hyperparameter Tuning

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
param_grid = {"C":np.logspace(-2,3,500), "penalty":['l1','l2'], 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 'max_iter':[200]}

tuned_logit = RandomizedSearchCV(classifier_3, param_grid , cv=skf, random_state=17)
tuned_logit.fit(x_train,y_train)

tuned_logit.best_params_, tuned_logit.best_score_

#tuned_logit.C_

p = tuned_logit.predict(x_test)

accuracy_score(y_test, p)

q = tuned_logit.predict_proba(x_test)

lr_proba_2 = q[:,1]

# Defining probability thresholds to use between 0 and 1
#prob_thres = np.linspace(0,1,num=100)

x_test_pred_2=[]
  
for l in lr_proba_2:
  if l>0.50:
   x_test_pred_2.append(1)
  if l<0.50:
   x_test_pred_2.append(0)

accuracy_score(y_test, x_test_pred_2)

print(classification_report(y_test, x_test_pred_2))

