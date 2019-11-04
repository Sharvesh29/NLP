# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/Practice Dumps/Machine Learning/Machine Learning A-Z udemy/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
## First clean one line of a dataset and then apply FOR loop condition for all others.
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
##Applyin loop for all 1000 Reviews
for i in range(0, 1000):
    ##To Remove unwanteed Punctuations we use sub, but what we dont wanna remove are the alphabets so a-z & A-Z
    #the unwanted Punctutaions are replaced by Blank spaces
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    ##Convering all to Lower Case
    review = review.lower()
    ##Importing nltk and  stopwords to remove unwanted preps or articles
    #Spit function spilts our rreview into list and we put a loop thrugh it to eliminate  STOP WORDS
    review = review.split()
    ##Stemming Process to revert words back to its ROOT *loved = love
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    ##Joinning the List into one word with balnk sapces btwn each word
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##RESULTS
# CONFUSION MATRIX
# 55	42
# 12	91

#So our model made (55+91)= 146 Correct Predictions
#and (42+12) = 54 Incorrect Prredictions
#ACCURACY OF Correct Prediction = 146/200 = 0.73 or 73%
#Since we trained only 800 reviews in our trainning set our accuracy is low.
#Higher Trainning  sets = Higher Accuraacy as NAIVE BAYES classifier finds strong correlating values.