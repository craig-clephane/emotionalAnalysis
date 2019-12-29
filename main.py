import pandas as pd
import numpy as np


dataname = 'emotioncleaned.csv'
data = pd.read_csv(dataname)

'''
    Application Paramters, can be changed accordingly
'''
NUMOFSAMPLES_PEREMOTION = 100
TFID_MODEL = False
BAGOFWORDS_MODEL = True
SKLEARN = False
KERAS = True

'''
    Classifcation Models
'''
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sentiment = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']
classifcationModelName = [
    "Sklearn Random Forest Classifcation",
    "Sklearn Naive Bayes Classifcation",
    "Sklearn SVM Classifcation",
    "Logistic Regression"
]
classifers = [
    RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=50),
    naive_bayes.MultinomialNB(),
    svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
    LogisticRegression()
]


def main():
    '''
        PHASE 1: DATAPREPARTION
    ''' 
    DF_List = list()
    Bigdata = pd.DataFrame()
    #Extract only the sentiment and content from the dataset
    rowData = data.loc[:, ['sentiment', 'content']]
    #Split the dataset into multiple dataframes each containing their emotions with the sample size set previously
    for sen in sentiment:
        rows = rowData.loc[rowData['sentiment'] == sen]
        DF_List.append(rows.iloc[0:NUMOFSAMPLES_PEREMOTION, :])
    #Append upon one dataset
    for dataset in DF_List:
        Bigdata = Bigdata.append(dataset)
    #Print Size of each emotion 
    print(Bigdata['sentiment'].value_counts())

    X = Bigdata['content']
    y = Bigdata['sentiment']
    '''
        PHASE 2: PREPROCESSING
    '''
    documents = preProcessingPhase(X)
    
    '''
        PHASE 3: FEATURE EXTRACTION
    '''
    #Split the dataset into sizes
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(documents, y, test_size=0.33)
    
    #Choose Feature extraction methods
    if BAGOFWORDS_MODEL == True:
        XTRAIN, XTEST = bagOfWordsModel(X_train,X_test)
    if TFID_MODEL == True:
        XTRAIN, XTEST = TfidModel(X_train,X_test)

    '''
        PHASE 4: CLASSIFCATION
    '''
    if SKLEARN == True:
        sklearnclassifcation(classifers, classifcationModelName, XTRAIN, y_train, XTEST, y_test)
    if KERAS == True:
        deeplearning(X_train, y_train, X_test, y_test)


def deeplearning(X_train, y_train, X_test, y_test):
    import os
    os.environ['KERAS_BACKEND']='theano'
    from keras.models import Sequential
    from keras import layers
    from keras.preprocessing.text import Tokenizer

    print(len(X_train))
    print(len(X_test))
    print(len(y_train))
    print(len(y_test))


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    vocab_size = len(tokenizer.word_index) + 1

    from keras.preprocessing.sequence import pad_sequences
    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.summary()
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    print("fitting model")
    history = model.fit(X_train, y_train)
    print("Finished fitting")

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def sklearnclassifcation(classifers, classifcationModelName, XTRAIN, YTRAIN, XTEST, YTEST):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    ind = 0
    for classifcationModel in classifers:
        classifcationModel.fit(XTRAIN, YTRAIN)
        y_pred = classifcationModel.predict(XTEST)
        print("\n"+str(classifcationModelName[ind]) +"\nAccuracy : " + str(accuracy_score(YTEST, y_pred)))
        ind=ind+1

def preProcessingPhase(X):
    from nltk.stem import WordNetLemmatizer
    import re
    from nltk.corpus import stopwords

    stemmer = WordNetLemmatizer()
    documents = []
    for sen in X:
        
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(sen))
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
    return documents

def bagOfWordsModel(X_train, X_test):
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer()
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)
    return X_train_dtm, X_test_dtm

def TfidModel(X_train, X_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    tfidfconverter = TfidfVectorizer(max_features= 1500,stop_words=stopwords.words('english'))
    X_train_tfid = tfidfconverter.fit_transform(X_train).toarray()
    X_test_tfid = tfidfconverter.fit_transform(X_test).toarray()
    return X_train_tfid, X_test_tfid


if __name__ == '__main__':
    main()









def readdataToCSV():
    '''
        Function to extract tweets from a specific file
    '''
    NUMOFDATA = 50000
    i = 0
    data = pd.read_csv(dataname)
    df = pd.DataFrame(columns=['sentiment', 'content'])
    print("Start process")
    for row in data.itertuples():
        print(i)
        text = row[1]
        listings = text.split(',')
        df = df.append({'sentiment' :listings[2], 'content' :listings[1] }, ignore_index=True)
        i = i+1
        if i == NUMOFDATA:
            print("Finish process")
            break
    df.to_csv('emotioncleaned.csv')
    print("Saved")

#readdataToCSV()
