import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords

def countLabels(data):
    labels = []
    for lab in data.itertuples():
        t = lab[2]
        if t not in labels:
            labels.append(t)
    return labels

def dataSetUp(data, NUMOFSAMPLES_PEREMOTION, LIMIT_OF_EMOTION):
	DF_List = list()
	df = pd.DataFrame()
	sentiment = countLabels(data)
	rowData = data.loc[:, ['sentiment', 'content']]
	for sen in sentiment:
		rows = rowData.loc[rowData['sentiment'] == sen]
		DF_List.append(rows.iloc[0:NUMOFSAMPLES_PEREMOTION, :])
	i = 0
	for dataset in DF_List:
		df = df.append(dataset)
		i = i+1
		if i is LIMIT_OF_EMOTION:
			break
	print(df['sentiment'].value_counts())
	X = df['content']
	y = df['sentiment']

	print('\nNumber of Data Rows', X.shape)
	print('Number of Label Rows', y.shape)
	return X, y

def preprocessDocuments(X):
	print("Preprocessing")
	stemmer = WordNetLemmatizer()
	documents = []
	for sen in X:
		document = re.sub(r'@[^\s]+','',str(sen))
		document = re.sub(r'\W', ' ', document)
		document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
		document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
		document = re.sub(r'\s+', ' ', document, flags=re.I)
		document = re.sub(r'^b\s+', '', document)
		document = document.lower()
		document = document.split()
		document = [stemmer.lemmatize(word) for word in document]
		document = ' '.join(document)
		documents.append(document)
	print("End Preprocessing")
	return documents
