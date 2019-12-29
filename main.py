import json
import pandas as pd
from pathlib import Path
import csv
import numpy as np
import preprocessing as prepro
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
dataname = '100sample.csv'
data = pd.read_csv(dataname, header=None)
np.random.seed(500) 
def evaluate(xtrain, xtest, ytrain, ytest):
    print("Xtrain size : " +str(len(xtrain)))
    print("Ytrain size : " +str(len(ytrain)))
    print("Xtest size : " +str(len(xtest)))
    print("Ytest size : " +str(len(ytest)))
    
def main():
    
    X = data[3]
    print(X.head)
    y = data[1]
    X = preprocessing(X)
    print(X.head)
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)
    evaluate(xtrain, xtest, ytrain, ytest)

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])

    text_clf = text_clf.fit(xtrain, ytrain)
    predicted = text_clf.predict(xtest)
    print(np.mean(predicted == ytest))

    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, random_state=42)),
    ])

    text_clf_svm = text_clf_svm.fit(xtrain, ytrain)
    predicted_svm = text_clf_svm.predict(xtest)
    print(np.mean(predicted_svm == ytest))



def main2():
    import re
    from nltk.corpus import stopwords
    X = data[3]
    print(X.head)
    y = data[1]
    documents = []

    from nltk.stem import WordNetLemmatizer

    stemmer = WordNetLemmatizer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        
        # Converting to Lowercase
        document = document.lower()
        
        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        
        documents.append(document)
    print(documents)
    # X = data[3]
    # print(X.head)
    # y = data[1]
    # X = preprocessing(X)
    # print(X.head)
    # labels = LabelEncoder()
    # y = labels.fit_transform(y)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = tfidfconverter.fit_transform(documents).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))



def preprocessing(preData):
    indexValue = 0 
    #Preprocessing 
    print("Preprocessing")
    for row in preData:
        preData[indexValue] = prepro.initPreProcessingText(row)
        indexValue +=1
    print("Preprocessing Finished")
    return preData


main2()