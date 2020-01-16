# Author: Craig Clephane
# Last Edited 15/01/2020
# Summary : Script which conrtains the functions for feature extraction and anything related
# to converting the documents into readable values.

import os

# Method Toggles ------------------- #'Token', 'Bag', 'TFID'
method = 'Token'
	
# Feature Extraction Function, decides at run time which feature extraction method the
# program will be using
def featureExtractionMethod(X, y, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
	if method is 'Token':
		X, vocab_size, word_index = tokenising_words(X, MAX_NB_WORDS)
		X = padding_words(X, MAX_SEQUENCE_LENGTH)
		Y = label_encoding(y)
		return X, Y, vocab_size, word_index
	return X, Y

# Coverts words into indices and returns the documents, vocab size and a word index used for embedding
def tokenising_words(documents, MAX_NB_WORDS):
	from keras.preprocessing.text import Tokenizer
	print("\nPreparing to convert words to indices")
	tokenizer = Tokenizer(MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(documents)
	X = tokenizer.texts_to_sequences(documents)
	print("Tokenizer Successful")
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	vocab_size = len(tokenizer.word_index) + 1
	return X, vocab_size, word_index

# Returns each input entry with the max length passed by argument
def padding_words(X, MAX_SEQUENCE_LENGTH):
	from keras. preprocessing.sequence import pad_sequences
	print("\nPreparing to Pad Data")
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	print("Padded data Successfully")
	print('Shape of data:', str(X.shape) + "\n")
	return X

# Assigns each unique label (emotion) to an integer value
def label_encoding(y):
	import pandas as pd
	print("\nPreparing to perform label encoding")
	Y = pd.get_dummies(y)
	print("label encoding Successful")
	print('Shape of labels:', str(Y.shape) + "\n")
	return Y

# Build a word embedding which is a vectorised representation of words. 
# Function uses trained GloVe models for word embeddings. Returns an embedding layer
# use for neural networks.
def wordEmbeddings(word_index, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
	from keras.layers import Embedding
	import numpy as np
	print("\nLoading Glove Vectors...")

	embeddings_index = {} 
	f = open(os.path.join('', 'glove.6B.100d.txt'), 'r', encoding="utf-8")
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
