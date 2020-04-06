# Author: Craig Clephane
# Last Edited 16/01/2020
# Summary : Main Script


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import preprocessing as preprocessing
import featureExtraction as feature
import numpy as np 
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import modelCreation as modelCreation
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
##from keras.preprocessing.text import text_to_sequences


from sklearn.neural_network import MLPClassifier


def features(feature, X):
	from keras_preprocessing.text import Tokenizer
	from keras_preprocessing.text import hashing_trick
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer

	## This class allows to vectorize a text corpus, by turning each text into either 
	## a sequence of integers (each integer being the index of a token in a dictionary) 
	## or into a vector where the coefficient for each token could be binary,
	## based on word count, based on tf-idf.

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(X)

	##Tokenising words / Converting words to indices
	word_indices = tokenizer.texts_to_sequences(X)

	vocab_size = len(tokenizer.word_index)+1

	print("Vocab size : " + str(vocab_size))
	#print(tokenizer.word_counts)
	#print(tokenizer.document_count)

	word_index = tokenizer.word_index

	#print(word_index)
	#print(tokenizer.word_docs)

	if feature == 'token':
		##Pad the input entry with 20 words each
		x_data=pad_sequences(word_indices, padding='post', maxlen=MAX_SEQUENCE_LENGTH)
		print(X[2])
		print(x_data[2])

	elif feature == 'bag':
		vectorizer = CountVectorizer()
		vectorizer.fit(X)
		print(vectorizer.vocabulary_)
		x_data = vectorizer.transform(X)
		print(x_data.shape)
		print(type(x_data))
		print(x_data.toarray())

	elif feature == 'tfid':
		vectorizer = TfidfVectorizer(use_idf=True)
		vectorizer.fit(X)
		x_data = vectorizer.transform(X)

	
	return x_data, vocab_size, word_index


def modelcreate(model_name, vocab_size, xshape, yshape, word_index):


	if model_name == ("hybrid" or "conv" or "lstm"):

		print(model_name)

		tag = 'deep'

		from keras import layers
		from keras.models import Sequential
		from keras.layers.core import Activation, Dense, Masking
		from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, MaxPooling1D, Conv1D
		from keras.layers import LSTM, Bidirectional
		from keras.layers.embeddings import Embedding

		model = Sequential()
		#model.add(embeddings(word_index))
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

	elif model_name == ("tree") or ("mlp"):

		tag = 'tradtional'

		if model_name == 'tree':

			model = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=2, random_state=None,)

		elif model_name == 'mlp':

			model =  mlp = MLPClassifier(hidden_layer_sizes=(50,500,500,50),activation="relu", max_iter=1000, solver='adam', learning_rate_init=0.17)

	return model, tag


	#print("X_data shape : ", x_data.shape)
	#print("Y_data shape : ", y_data.shape)


def embeddings(word_index):
	from keras.layers import Embedding
	print("\nLoading Glove Vectors...")

	embeddings_index = {} 
	f = open(os.path.join('', 'glove.6B.50d.txt'), 'r', encoding="utf-8")
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

def evaluate(model, tag, x_data, y_data, batch_size, name_mapping):
	from sklearn import model_selection
	from sklearn.model_selection import train_test_split, cross_val_score

	X_train,X_test, Y_train, Y_test=train_test_split(x_data,y_data)
	print(X_train.shape)
	print(X_test.shape)
	print(Y_train.shape)
	print(Y_test.shape)

	if tag == 'deep':
		
		#epochs = [10,15,20,25,30,35,40,45,50]
		epochs = [130]
		x_valid, y_valid = X_train[:batch_size], Y_train[:batch_size]
		x_valid2, y_valid2 = X_train[:batch_size], Y_train[:batch_size]
		for ep in epochs:
		##checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
			#history = model.fit(x_valid2, y_valid2, validation_data=(x_valid, y_valid), epochs=ep, batch_size=batch_size)
			#history = model.fit(X_train, Y_train, epochs=ep, batch_size=batch_size,validation_split=0.1, verbose=0)
			history = model.fit(X_train, Y_train, epochs=ep, batch_size=batch_size, validation_data=(X_test, Y_test))
			print("Model finished fitting")

			scores = model.evaluate(X_test, Y_test)

			print('Test Accuracy:', scores[1])

			# k_fold_validation = model_selection.KFold(n_splits=5, random_state=12)
			# CVscore = cross_val_score(model, X_train, Y_train, cv=k_fold_validation, scoring='accuracy')
			# print("CV Score : " + str(CVscore.mean()))

		from sklearn.metrics import confusion_matrix
		print("confusion matrix test")
		print("X_test size" +str(len(X_test)))
		print("Y_test size" + str(len(Y_test[1])))

		y_pred = model.predict(X_test)
		print(name_mapping)
		print(metrics.confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1)))
		


		modelCreation.modelEval(history)
	


	elif tag == 'tradtional':
		model = model.fit(X_train,Y_train)
		y_pred = model.predict(X_test)

		print("Test Accuracy:",metrics.accuracy_score(Y_test, y_pred))

		k_fold_validation = model_selection.KFold(n_splits=5, random_state=12)
		CVscore = cross_val_score(model, X_train, Y_train, cv=k_fold_validation, scoring='accuracy')
		print("CV Score : " + str(CVscore.mean()))



def onehotcoding(y):

	from sklearn.preprocessing import LabelEncoder
	from keras.utils import np_utils
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(y)
	name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
	print(name_mapping)
	y_data=np_utils.to_categorical(integer_encoded)

	return y_data, name_mapping

FEATURE = 'token' ; batch_size = 300
#FEATURE = 'bag'; batch_size = 30
#FEATURE = 'tfid'; batch_size = 300

#MODELNAME = 'lstm'
#MODELNAME = 'conv'
MODELNAME = 'hybrid'
#MODELNAME = 'tree'
#MODELNAME = 'mlp'

dataname = 'emotionDataset1.csv'
#dataname = 'emotionDataset2.csv'

LIMIT_OF_EMOTION = 6
NUMOFSAMPLES_PEREMOTION = 1000
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 50

def main():

	data = pd.read_csv(dataname)
	
	Xb, y = preprocessing.dataSetUp(data, NUMOFSAMPLES_PEREMOTION, LIMIT_OF_EMOTION)

	X = preprocessing.preprocess(Xb)

	x_data, vocab_size, word_index = features(FEATURE, X)

	y_data, name_mapping = onehotcoding(y)

	model, tag = modelcreate(MODELNAME, vocab_size, x_data.shape[1], y_data.shape[1], word_index)

	evaluate(model, tag, x_data, y_data, batch_size, name_mapping)

main()