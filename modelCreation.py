# Author: Craig Clephane
# Last Edited 16/01/2020
# Summary : Model Creation

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
import featureExtraction as feature

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

LSTM = True
DENSE = False
CONV = False
test = False


activation = 'softmax'
optimizer = 'adam'

embedding = False

output_file = 'output.txt'

LOSS_FUNCTION = 'categorical_crossentropy'
	#'mean_squared_error',categorical_crossentropy

NUM_NEURONS = [64]

def returnModels(X, ysize,vocab_size, word_index, MAX_SEQUENCE_LENGTH):
	models = []
	modelDes = []
	if embedding is True:
		embedding_layer = feature.wordEmbeddings(word_index, 100, MAX_SEQUENCE_LENGTH)
	else:
		embedding_layer = 100
	for neurons in NUM_NEURONS:
		if LSTM is True:
			model = LSTMModel(neurons, 7, vocab_size, embedding_layer, X.shape[1], ysize)
			models.append(model)
		if test is True:
			model = testingmodel(neurons, 7, vocab_size, embedding_layer, X.shape[1], ysize)
			models.append(model)
		if DENSE is True:
			model = dense(neurons, 7, vocab_size, embedding_layer, X.shape[1], ysize)
			models.append(model)
		if CONV is True:
			model = conv(neurons, 7, vocab_size, embedding_layer, X.shape[1], ysize)
			models.append(model)
	return models

def modelString(neurons,activation,optimizer,Loss):
	modelDesc = ("Number of Neurons : " + str(neurons) + "\nActivation Function : " + str(activation) + "\nOptimizer : " + str(optimizer) + "\nLoss : " + str(Loss) + "\n")
	print(modelDesc)
	return modelDesc

def GridSearch(X, Y, X_train, X_test, Y_train, Y_test, vocab_size, epochs,batch_size):
	param_grid = dict(neurons=[32, 64, 128], kernel_size=[3, 5, 7], vocab_size=[vocab_size], embedding_dim=[100], maxlen=[X.shape[1]], ysize=[Y.shape[1]])
	model = KerasClassifier(build_fn=LSTMModel, epochs=epochs, batch_size=batch_size, verbose=False)
	grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
	grid_result = grid.fit(X_train, Y_train)
	test_accuracy = grid.score(X_test, Y_test)
	with open(output_file, 'a') as f:
		s = ('Best accuracy : '
			'{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
		output_string = s.format(
			grid_result.best_score_,
			grid_result.best_params_,
			test_accuracy)
		print(output_string)
		f.write(output_string)

def LSTMModel(neurons, kernel_size, vocab_size, embedding_layer, maxlen, ysize):

	from keras.layers import LSTM
	from keras import layers

	model = Sequential()
	if embedding is True:
		print("Adding Word Embeddings--------------------")
		model.add(embedding_layer)
	else:
		model.add(Embedding(vocab_size, embedding_layer, input_length=maxlen))
	model.add(LSTM(neurons, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(ysize, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning LSTM Model")
	return model

def dense(neurons, kernel_size, vocab_size, embedding_layer, maxlen, ysize):

	from keras import layers

	model = Sequential()
	if embedding is True:
		print("Adding Word Embeddings--------------------")
		model.add(embedding_layer)
	else:
		model.add(Embedding(vocab_size, embedding_layer, input_length=maxlen))
	model.add(layers.Flatten())
	model.add(Dense(neurons, activation=activation))
	model.add(Dense(ysize, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning dense Model")

	return model

def conv(num_filters, kernel_size, vocab_size, embedding_layer, maxlen, ysize):

	from keras import layers

	model = Sequential()
	if embedding is True:
		print("Adding Word Embeddings--------------------")
		model.add(embedding_layer)
	else:
		model.add(Embedding(vocab_size, embedding_layer, input_length=maxlen))
	model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
	model.add(layers.Flatten())
	model.add(Dense(10, activation=activation))
	model.add(Dense(ysize, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning conv Model")
	return model

def testingmodel(neurons, kernel_size, vocab_size, embedding_layer, maxlen, ysize):

	from keras.layers import LSTM, Flatten, Conv1D, MaxPooling1D
	from keras import layers

	model = Sequential()
	if embedding is True:
		print("Adding Word Embeddings--------------------")
		model.add(embedding_layer)
	else:
		model.add(Embedding(vocab_size, embedding_layer, input_length=maxlen))
	model.add(Conv1D(30, 1, activation="relu"))
	model.add(MaxPooling1D(4))
	model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(500,activation='relu'))
	model.add(Dense(300,activation='relu'))
	model.add(Dense(ysize, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning LSTM Model")
	return model


def modelEval(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
