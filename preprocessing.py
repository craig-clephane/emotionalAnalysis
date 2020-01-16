# Author: Craig Clephane
# Last Edited 15/01/2020
# Summary : Script which conrtains the functions for preprocessing

import pandas as pd
from nltk.stem import PorterStemmer
porter=PorterStemmer()
import re, unicodedata, string
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from string import digits

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Returns a list of labels
def countLabels(data):
    labels = []
    for lab in data.itertuples():
        t = lab[2]
        if t not in labels:
            labels.append(t)
    return labels

# Function to organise, and prepare the data in regards to the number of samples, and
# the limit of emotions. Returns the result of the datasetup function
def dataSetUp(data, NUMOFSAMPLES_PEREMOTION, LIMIT_OF_EMOTION):
	DF_List = list()
	localbool = True
	df = pd.DataFrame()
	sentiment = countLabels(data)
	rowData = data.loc[:, ['sentiment', 'content']]
	for sen in sentiment:
		rows = rowData.loc[rowData['sentiment'] == sen]
		DF_List.append(rows.iloc[0:NUMOFSAMPLES_PEREMOTION, :])
	i = 0
	count = 0
	val = []
	for dataset in DF_List:
		size = dataset.shape
		if size[0] == NUMOFSAMPLES_PEREMOTION:
			count = count + 1
			df = df.append(dataset)
		else:
			val.append(i)
		i = i +1
		if count is LIMIT_OF_EMOTION:
			localbool = False
			break

	if localbool is True:
		sizearray = []
		for c in val:
			dataset = DF_List[c]
			sizearray.append(dataset.shape[0])
		while len(sizearray) > 0:
			for dataset in DF_List:
				value = sizearray.index(max(sizearray))
				if dataset.shape[0] == sizearray[value]:
					df = df.append(dataset)
					sizearray.pop(value)
					count = count + 1
				if count is LIMIT_OF_EMOTION:

					return init_data_setup(df)
	return init_data_setup(df)

# Returns the chosen emotion sets with their labels
def init_data_setup(df):
	X = df['content']
	y = df['sentiment']
	print(df['sentiment'].value_counts())
	print('\nNumber of Data Rows', X.shape)
	print('Number of Label Rows', y.shape)
	return X, y

# Main function to call other preporcessing methods, returns an array of preporcessed documents
def preprocess(X):
	from textblob import TextBlob
	print("\nPreprocessing")
	documents = []

	X = removeStopwords(X)
	#X = spellingCorrection(X)

	for document in X:
		document = lowercasing(document)
		document = removeAt(strip_links(document))
		document = remove_values(document)
		document = remove_punctuation(document)
		document = removeNonAscii(document)
		document = removeLowLetterWords(document)
		document = stemSentence(document)
		document = removeBlankSpace(document)
		documents.append(document)
	IdentifyFreWords(documents)
	print("\nEnd Preprocessing")
	return documents


# ------------------ Individual preprocessing functions ------------------- #
# Each function performs an individual task important to preprocessing the documents
# before feature extraction. Functions which return X effect every document all at once,
# where as function return a single document performs preprocessing on one sentence at
# a time

# Removes stop words from the entire corpus
def removeStopwords(X):
	X = X.apply( lambda t : ' '.join ( word for word in t.split() if word not in stop_words))
	return X

# Support function to identify frequent words within the documents
def IdentifyFreWords(documents):
	word_count = Counter(documents)
	Frequent = word_count.most_common(20)
	listOfWords = []
	for item in Frequent:
		listOfWords.append(item[0])

# Converts each document into lowercase
def lowercasing(document):
	document = document.lower()
	return document

# perform spell checking on each document
def spellingCorrection(X):
	X = X.apply(lambda x: str(TextBlob(x).correct()))
	return X

# Remove twitter handle on each doucment
def removeAt(document):
	document = re.sub('@[^\s]+','',document)
	return document

def remove_values(document):
	pattern = '[0-9]'
	document = ''.join([i for i in document if not i.isdigit()])
	return document

# Remove links to webpages on each document
def strip_links(document):
	link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
	links = re.findall(link_regex, document)
	for link in links:
		document = document.replace(link[0], ', ')
	return document

# Remove punctuation on each document 
def remove_punctuation(document):
	token_words=word_tokenize(document)
	new_words=[]
	for word in token_words:
		new_word = re.sub(r'[^\w\s]', '', word)
		if new_word != '':
			new_words.append(new_word)
	document = ' '.join(new_words)
	return document


def removeNonAscii(document):
	token_words=word_tokenize(document)
	new_words=[]
	for word in token_words:
		new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		new_words.append(new_word)
	document = ' '.join(new_words)
	return document

# Remove low letter words, less than or equal to two from each document
def removeLowLetterWords(document):
	document = re.sub(r'\b\w{1,2}\b', '', document)
	return document

# Performs stemming on each document
def stemSentence(document):
	token_words=word_tokenize(document)
	stem_sentence=[]
	for word in token_words:
		stem_sentence.append(porter.stem(word))
		stem_sentence.append(" ")
	return "".join(stem_sentence)

# Performs lematzing on each document
def lemnatizeSentence(document):
	token_words=word_tokenize(document)
	lem_sentence=[]
	for word in token_words:
		lem_sentence.append(lemmatizer.lemmatize(word))
		lem_sentence.append(" ")
	return "".join(lem_sentence)

# Removes any addional blank space within each document
def removeBlankSpace(document):
	document = re.sub(' +', ' ', document)
	document = document.lstrip(' ')
	return document 

