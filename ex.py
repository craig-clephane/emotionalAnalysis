## Author: Craig Clephane
## Last Edited 09/04/2020
## Summary : Main Script

## HELP -------------------------------------- 
## Double Hashtag used for comment.
## Single hashtag used for removeable code.
## -------------------------------------------

## Imported scripts used in this project.
import preprocessing as preprocessing
import interface as interface

## Imported modules used throught these program.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np 
import pandas as pd
import os
import modelCreation as modelCreation

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


## This class allows to vectorize a text corpus, by turning each text into either 
## a sequence of integers (each integer being the index of a token in a dictionary) 
## or into a vector where the coefficient for each token could be binary,
## based on word count, based on tf-idf.
## Input : A string with the feature name and an array of documents.
## Output : Document of features, the vocabulary size and the index to each word inside the documents
def features(feature, X):

	from keras_preprocessing.text import Tokenizer
	from keras_preprocessing.text import hashing_trick
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer

	## Tokenising words / Converting words to indices
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(X)
	word_indices = tokenizer.texts_to_sequences(X)

	## Extracts the vocab size and the word indexes from the tokenizer 
	vocab_size = len(tokenizer.word_index)+1
	word_index = tokenizer.word_index

	## Printed values used for debuging and viewing information
	print("\n----------------------\nFEATURE DETAILS")
	print("Vocab size : " + str(vocab_size))
	print("Document Count : " + str(tokenizer.document_count))

	## Pad the input entry with 20 words each
	if feature == 'token':
		x_data=pad_sequences(word_indices, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

	elif feature == 'bag':
		vectorizer = CountVectorizer()
		vectorizer.fit(X)
		x_data = vectorizer.transform(X)

	elif feature == 'tfid':
		vectorizer = TfidfVectorizer(use_idf=True)
		vectorizer.fit(X)
		x_data = vectorizer.transform(X)

	return x_data, vocab_size, word_index

## Create model function, used to return a model and a tag deseried by the user with the predetermed variables.
## Creates predifened model based on the type of model chosen by the user.
## Input : A string with the model name, a vocab size from the feature extraction function, the size of x data, the size of y data
## and the word index array. 
## Output : Created tensorflow model and a string stating the type of model. 
def modelcreate(model_name, vocab_size, xshape, yshape, word_index):

	##  Selected if the model is a deep or hybrid learning model. 
	if model_name == ("hybrid") or ("conv") or ("lstm"):
		from keras import layers
		from keras.models import Sequential
		from keras.layers.core import Activation, Dense, Masking
		from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, MaxPooling1D, Conv1D
		from keras.layers import LSTM, Bidirectional
		from keras.layers.embeddings import Embedding

		print("Model chosen : " + str(model_name))
		tag = 'deep'

		model = Sequential()
		em, tg = interface.embedding()

		if tg is True:
			model.add(embeddings(word_index, em))
		else:
			model.add(Embedding(input_dim=vocab_size, output_dim= 50, input_length=xshape))
		
		if model_name == 'conv':
			model.add(Conv1D(50, 3, activation='relu', strides=1, padding='valid'))
			model.add(layers.Flatten())
			model.add(Dense(100, activation='relu'))
			model.add(Dense(yshape, activation='relu'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		elif model_name == 'lstm':
			model.add(LSTM(50, dropout=0.1, recurrent_dropout=0.1))
			model.add(Dense(100,activation='relu'))
			model.add(Dense(50,activation='relu'))
			model.add(Dense(yshape, activation='relu'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		elif model_name == 'hybrid':
			model.add(Conv1D(50, 3, activation='relu', strides=1, padding='valid'))
			model.add(MaxPooling1D(4))
			model.add(LSTM(500, dropout=0.1, recurrent_dropout=0.1))
			model.add(Dense(500,activation='relu'))
			model.add(Dense(300,activation='relu'))
			model.add(Dense(yshape, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
		print(model.summary())

	## Selected if model is tradional machine learning
	elif model_name == ("tree"):
		tag = 'tradtional'
		if model_name == 'tree':
			model = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=2, random_state=None)

	return model, tag

## Embedding function, used to create an embedding layer with the passed embedding text file name.
## Input : word index array and the name of the file for the word embeddings.
## Output : Document of features, the vocabulary size and the index to each word inside the documents.
def embeddings(word_index, name):
	from keras.layers import Embedding
	print("\nLoading Glove Vectors...")

	embeddings_index = {} 
	f = open(os.path.join('', name), 'r', encoding="utf-8")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close

	print('Loaded GloVe Vectors Successfully')
	print('Performing word embedding')

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	print("Embedding Matrix Generated : ", str(embedding_matrix.shape) + "\n")

	embedding_layer = Embedding(len(word_index) + 1,EMBEDDING_DIM, weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=False)

	return embedding_layer

## Evaluate model function. 
## Determines the type of model, either deep learning or tradional learning. 
## Spilts the dataset into training and testing, then trains the model using the training set.
## Testing data is used against the model for each model as a validation set to determine the performance. 
## An overall accracy is scored at the end of the epoch cycle. 
## Confusion matrix is then established to show how the model performs. 
## Input : String containing a model name, string containing type of model, X data set, Y data set, batch size int, a map of the emotions used for evaulation.
## Output : None
def evaluate(model, tag, x_data, y_data, batch_size, name_mapping):
	from sklearn import model_selection
	from sklearn.model_selection import train_test_split, cross_val_score

	X_train,X_test, Y_train, Y_test=train_test_split(x_data,y_data)
	print(X_train.shape)
	print(X_test.shape)
	print(Y_train.shape)
	print(Y_test.shape)

	if tag == 'deep':
		
		epochs = [130]
		x_valid, y_valid = X_train[:batch_size], Y_train[:batch_size]
		x_valid2, y_valid2 = X_train[:batch_size], Y_train[:batch_size]
		for ep in epochs:
			history = model.fit(X_train, Y_train, epochs=ep, batch_size=batch_size, validation_data=(X_test, Y_test))
			print("Model finished fitting")
			scores = model.evaluate(X_test, Y_test)
			print('Test Accuracy:', scores[1])

		from sklearn.metrics import confusion_matrix
		print("confusion matrix test")
		print("X_test size" +str(len(X_test)))
		print("Y_test size" + str(len(Y_test[1])))

		y_pred = model.predict(X_test)
		print(name_mapping)
		print(metrics.confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1)))
		
		modelCreation.modelEval(history)
	
	## Training data is used to train the decision tree, then testing data is used to determine the accuracy. 
	elif tag == 'tradtional':
		model = model.fit(X_train,Y_train)
		y_pred = model.predict(X_test)

		print("Test Accuracy:",metrics.accuracy_score(Y_test, y_pred))
		k_fold_validation = model_selection.KFold(n_splits=5, random_state=12)
		CVscore = cross_val_score(model, X_train, Y_train, cv=k_fold_validation, scoring='accuracy')
		print("CV Score : " + str(CVscore.mean()))


## The labels are turned into intergers, or one got coding labels as machine learning models use intergers rather than strings.
## Input : Y vector of labels (each emotion).
## Output : Y vector of labels as integrers with a mapping of each emotion to the corresponding emotion. 
def onehotcoding(y):

	from sklearn.preprocessing import LabelEncoder
	from keras.utils import np_utils
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y)
	name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
	print("Emotional Mapping : " + str(name_mapping))
	y_data=np_utils.to_categorical(integer_encoded)

	return y_data, name_mapping

## Static variables
batch_size = 300
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 50

## Main function initally ran at run time. Performs each step sequentially. 
## Input : None.
## Output : None
def main():

	data = pd.read_csv(interface.dataFileSelection())

	preprocessing.details(data)

	Xb, y = preprocessing.dataSetUp(data, interface.numberOfSamples(), interface.limitOfEmotions())

	X = preprocessing.preprocess(Xb)

	x_data, vocab_size, word_index = features(interface.featureSelection(), X)

	y_data, name_mapping = onehotcoding(y)

	model, tag = modelcreate(interface.modelSelection(), vocab_size, x_data.shape[1], y_data.shape[1], word_index)

	evaluate(model, tag, x_data, y_data, batch_size, name_mapping)

main()