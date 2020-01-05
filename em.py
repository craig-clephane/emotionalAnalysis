import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing as preprocessing
import featureExtraction as feature
import modelCreation as modelCreation

from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

#dataname = 'emotionDataset1.csv'
dataname = 'emotionDataset2.csv'
LIMIT_OF_EMOTION = 10
NUMOFSAMPLES_PEREMOTION = 1000
TFID_MODEL = True
BAGOFWORDS_MODEL = False
output_file = 'output.txt'

SKLEARN = False
KERAS = True

epochs = 20
batch_size = 128

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250

def main():

	data = pd.read_csv(dataname)

	X, y = preprocessing.dataSetUp(data, NUMOFSAMPLES_PEREMOTION, LIMIT_OF_EMOTION)

	X = preprocessing.preprocessDocuments(X)

	X, Y, vocab_size = feature.featureExtractionMethod(X, y, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state = 42)

	#models, modelDescList = modelCreation.returnModels(X_train, Y.shape[1], MAX_NB_WORDS)

	#print("Number of Models : " + str(len(models)))

	print(X_train.shape,Y_train.shape)
	print(X_test.shape,Y_test.shape)
	
	param_grid = dict(num_filters=[32, 64, 128], kernel_size=[3, 5, 7], vocab_size=[vocab_size], embedding_dim=[100], maxlen=[X.shape[1]])
	model = KerasClassifier(build_fn=modelCreation.LSTMModel, epochs=epochs, batch_size=batch_size, verbose=False)
	grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5)
	grid_result = grid.fit(X_train, Y_train)
	test_accuracy = grid.score(X_test, Y_test)

	# for model in models:
	# 	print("\nFitting Model "+str(i+1)+"\n"+ modelDescList[i] + "\nEpochs : " + str(epochs) + "\nBatch Size : " + str(batch_size))
	# 	history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
	# 	print("Model finished fitting")
	# 	modelCreation.modelEval(history)
	# 	i = i +1

	with open(output_file, 'a') as f:
		s = ('Best accuracy : '
			'{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
		output_string = s.format(
			grid_result.best_score_,
			grid_result.best_params_,
			test_accuracy)
		print(output_string)
		f.write(output_string)

main()
