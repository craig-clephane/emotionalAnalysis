from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

method = 'Token'
	#'Token', 'Bag', 'TFID'


def featureExtractionMethod(X, y, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
	if method is 'Bag':
		X, Y = bagOfWordsModel(X, y)
	if method is 'Token':
		X, Y, vocab_size = tokenizer(X, y,MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
		return X, Y, vocab_size
	return X, Y

def bagOfWordsModel(X, y):
    vect = CountVectorizer()
    X = vect.fit_transform(X)
    Y = labelTensor(y)
    return X, Y

def TfidModel(X_train, X_test):
    tfidfconverter = TfidfVectorizer(max_features= 1500,stop_words=stopwords.words('english'))
    X_train_tfid = tfidfconverter.fit_transform(X_train).toarray()
    X_test_tfid = tfidfconverter.fit_transform(X_test).toarray()
    return X_train_tfid, X_test_tfid

def tokenizer(documents, y, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
	from keras.preprocessing.text import Tokenizer
	from keras. preprocessing.sequence import pad_sequences

	tokenizer = Tokenizer(MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
	tokenizer.fit_on_texts(documents)
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
	X = tokenizer.texts_to_sequences(documents)
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
	print('Shape of data tensor:', X.shape)
	Y = labelTensor(y)
	vocab_size = len(tokenizer.word_index) + 1
	return X, Y, vocab_size

def labelTensor(y):
	Y = pd.get_dummies(y)
	print('Shape of label tensor:', Y.shape)
	return Y
