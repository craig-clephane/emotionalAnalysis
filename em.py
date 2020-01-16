# Author: Craig Clephane
# Last Edited 16/01/2020
# Summary : Main Script

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing as preprocessing
import featureExtraction as feature
import modelCreation as modelCreation

from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

dataname = 'emotionDataset1.csv'
#dataname = 'emotionDataset2.csv'
LIMIT_OF_EMOTION = 6
NUMOFSAMPLES_PEREMOTION = 4000

SKLEARN = False
KERAS = True

epochs = 10
batch_size = 124

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 20

def main():

	data = pd.read_csv(dataname)

	Xb, y = preprocessing.dataSetUp(data, NUMOFSAMPLES_PEREMOTION, LIMIT_OF_EMOTION)
	X = preprocessing.preprocess(Xb)

	#show(Xb, X, y)

	X, Y, vocab_size, word_index = feature.featureExtractionMethod(X, y, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
	
	#embedding_layer = feature.wordEmbeddings(word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state = 42)

	models = modelCreation.returnModels(X_train, Y.shape[1], vocab_size, word_index, MAX_SEQUENCE_LENGTH)

	print("Number of Models : " + str(len(models)))

	print(X_train.shape,Y_train.shape)
	print(X_test.shape,Y_test.shape)
	from sklearn.metrics import classification_report

	i = 0

	for model in models:
		print("\nFitting Model "+str(i+1)+"\n" + "\nEpochs : " + str(epochs) + "\nBatch Size : " + str(batch_size))
		history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
		#history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
		print("Model finished fitting")
		scores = model.evaluate(X_test, Y_test, verbose=0)
		print('Test Accuracy:', scores[1])
		modelCreation.modelEval(history)
		i = i +1

	#modelCreation.GridSearch(X, Y, X_train, X_test, Y_train, Y_test, vocab_size,epochs, batch_size)

def show(Xb, X, y):
	for i, (old, new, label) in enumerate(zip(Xb, X, y)):
		print("\n" + label + "\nBefore: '" +  old +"'\nAfter: " + "'"+ new+ "'")

main()
