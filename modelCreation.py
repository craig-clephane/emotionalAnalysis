from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt

LSTM = True
DENSE = False
CONV = False

EMBEDDING_DIM = 100

LOSS_FUNCTION = ['categorical_crossentropy']
	#'mean_squared_error',categorical_crossentropy

NUM_NEURONS = [250]


def returnModels(X, ysize, MAX_NB_WORDS):
	models = []
	modelDes = []
	for Loss in LOSS_FUNCTION:
		for neurons in NUM_NEURONS:
			if LSTM is True:
				model, modelDesc = LSTMModel(X, MAX_NB_WORDS, EMBEDDING_DIM, neurons, Loss, ysize)
				models.append(model)
				modelDes.append(modelDesc)
			if DENSE is True:
				model, modelDesc = dense(X, MAX_NB_WORDS, EMBEDDING_DIM, neurons, Loss, ysize)
				models.append(model)
				modelDes.append(modelDesc)
			if CONV is True:
				model, modelDesc = conv(X, MAX_NB_WORDS, EMBEDDING_DIM, neurons, Loss, ysize)
				models.append(model)
				modelDes.append(modelDesc)

	return models, modelDes

def modelString(neurons,activation,optimizer,Loss):
	modelDesc = ("Number of Neurons : " + str(neurons) + "\nActivation Function : " + str(activation) + "\nOptimizer : " + str(optimizer) + "\nLoss : " + str(Loss) + "\n")
	print(modelDesc)
	return modelDesc

# def LSTMModel(X, ma, em, n, l, y):

# 	from keras.layers import LSTM

# 	activation = 'softmax'
# 	optimizer = 'adam'

# 	model = Sequential()
# 	model.add(Embedding(ma, em, input_length=X.shape[1]))
# 	model.add(LSTM(n, dropout=0.2, recurrent_dropout=0.2))
# 	model.add(Dense(y, activation=activation))
# 	model.compile(loss=l, optimizer=optimizer, metrics=['accuracy'])
# 	print("\nReturning LSTM Model")
# 	modelDesc = modelString(n,activation,optimizer,l)

# 	return model, modelDesc

def LSTMModel(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

	from keras.layers import LSTM

	activation = 'softmax'
	optimizer = 'adam'

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
	model.add(LSTM(num_filters, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(6, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning LSTM Model")
	#modelDesc = modelString(n,activation,optimizer,l)

	return model

def dense(X, ma, em, n, l, y):

	from keras import layers

	activation = 'softmax'
	optimizer = 'adam'
	n_cols = X.shape[1]

	model = Sequential()
	model.add(Embedding(ma, em, input_length=X.shape[1]))
	model.add(layers.Flatten())
	model.add(Dense(n, activation=activation))
	model.add(Dense(y, activation=activation))
	model.compile(loss=l, optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning dense Model")
	modelDesc = modelString(n,activation,optimizer,l)

	return model, modelDesc


def modelEval(history):
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def conv(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

	from keras import layers

	activation = 'softmax'
	optimizer = 'adam'

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
	model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
	model.add(layers.Flatten())
	model.add(Dense(10, activation=activation))
	model.add(Dense(6, activation=activation))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	print("\nReturning dense Model")
	return model


# def conv(X, ma, em, n, l, y):

# 	from keras import layers

# 	activation = 'softmax'
# 	optimizer = 'adam'
# 	n_cols = X.shape[1]

# 	model = Sequential()
# 	model.add(Embedding(ma, em, input_length=X.shape[1]))
# 	model.add(layers.Conv1D(n, 5, activation='relu'))
# 	model.add(layers.Flatten())
# 	model.add(Dense(n, activation=activation))
# 	model.add(Dense(y, activation=activation))
# 	model.compile(loss=l, optimizer=optimizer, metrics=['accuracy'])
# 	print("\nReturning dense Model")
# 	modelDesc = modelString(n,activation,optimizer,l)

# 	return model, modelDesc
