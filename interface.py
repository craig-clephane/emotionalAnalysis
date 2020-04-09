## Author: Craig Clephane
## Last Edited 09/04/2020
## Summary : interface Script

## HELP -------------------------------------- 
## Double Hashtag used for comment.
## Single hashtag used for removeable code.
## -------------------------------------------

## ABOUT ------------------------------------
## Handful of functions called to return a specific value.
## functions include try and except blocks to avoid errors.
## ------------------------------------------

## Imported Modules used throughout functions.
import glob

## Prints CSV files in the directory, allows user to choose which file they wish to use.
## Loops if invalid value is presented.
## Input : None.
## Output : A single string containing a file name.
def dataFileSelection():
	print("\n----------------------------------\nChoose Datafile used for model training\n")
	files = glob.glob("*.csv")
	i = 1
	for fi in files:
		print(str(i) + " - " + str(fi))
		i+=1

	while(True):
		requested_file = input("\nEnter:")
		try: 
			if requested_file in files:
				return requested_file
			else:
				raise ValueError("Error: File not found, try again")
		except ValueError as ivf:
			print(ivf)
	
## Prints several types of models, allows user to choose.
## Loops if invalid value is presented.
## Input : None.
## Output : A single string containing a model name.
def modelSelection():
	print("\n----------------------------------\nChoose Model used for training\n")
	models = ['Convoultional Neural Network', 'Long Short Term Memory', 'Hybrid Neural Network', 'Decision Tree Classifer']
	modelid = ['conv', 'lstm', 'hybrid', 'tree']
	i = 0
	for modid in modelid:
		print(str(models[i]) + " - " + str(modid))
		i+=1

	while(True):
		requested_model = input("\nEnter:")
		try:
			if requested_model in modelid:
				return requested_model
			else:
				raise ValueError("Error: Enter a valid model, try again")
		except ValueError as ivf:
			print(ivf)

## Prints several types of feature extraction methods, allows user to choose.
## Loops if invalid value is presented.
## Input : None.
## Output : A single string containing the feature extraction name.
def featureSelection():
	print("\n----------------------------------\nChoose feature selection\n")
	feature = ['Keras Tokenizer', 'Bag of words', 'TFIDF']
	featureid = ['token', 'bag', 'tfid']
	i = 0 
	for feaid in featureid:
		print(str(feature[i]) + " - " + str(feaid))
		i+=1

	while(True):
		feature_request = input("\nEnter:")
		try:
			if feature_request in featureid:
				return feature_request
			else:
				raise ValueError("Error: Enter a valid feature extraction method, try again")
		except ValueError as ivf:
			print(ivf)

## Allows user to select the number of samples in each emotional dataset.
## Loops if invalid value is presented.
## Input : None. 
## Output : A single int with the number of samples per each emotion.
def numberOfSamples():
	print("\n----------------------------------\nChoose number of samples\n")

	while(True):
		num_of_samples = input("Enter:")
		try:
			if int(num_of_samples) > 1:
				return int(num_of_samples)
			else:
				raise ValueError("Error: Enter a valid value")
		except ValueError as ivf:
			print(ivf)	

## Allows user to select the number of emotions in the training and testing dataset.
## Loops if invalid value is presented. 
## Input : None.
## Output : A single int with the number of emotions being classified.
def limitOfEmotions():
	print("\n----------------------------------\nChoose number of emotions\n")

	while(True):
		num_of_emotions = input("Enter:")
		try:
			if int(num_of_emotions) > 1:
				return int(num_of_emotions)
			else:
				raise ValueError("Error: Enter a valid value")
		except ValueError as ivf:
			print(ivf)	

## Prints txt files in the directory, allows user to choose which file they wish to use.
## Loops if invalid value is presented.
## Input : None.
## Output : A single string containing a either a file name or the value "None".
def embedding():
	print("\n----------------------------------\nSelect Embedding Layer\n")
	files = glob.glob("*.txt")
	i = 1
	for fi in files:
		print(str(i) + " - " + str(fi))
		i+=1
	print(str(i) + " - " + str("None"))

	while(True):
		requested_file = input("\nEnter:")
		try: 
			if requested_file in files:
				return requested_file, True
			if requested_file == "None":
				return "None", False
			else:
				raise ValueError("Error: File not found, try again")
		except ValueError as ivf:
			print(ivf)