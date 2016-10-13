'''
Author: Mohit Mangal
Email: mohit.mangal2@gmail.com, mohit.mangal@lnmiit.ac.in
Description: Following file classifies questions given in
             test.csv into what, when, affirmation and unknown
             classes.
             It considers questions given in files what,when and
             affirmation as training data corresponding to each
             type.
             output is written to testOut.csv file
'''
import nltk
import sklearn
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import csv

# Reading training dataset
distinctSymbols = []

whatPattern = []
whenPattern = []
affirmationPattern = []

fp = open("affirmation","r")
for row in fp:
	row = row.lower()
	row = row.split(',')[-1]
	posTags = nltk.pos_tag(row.lower().strip().split())
	affirmationPattern += [[posTags[0][0], '!!!!!@@@@@@', posTags[1][1]]]
	if posTags[0][0] not in distinctSymbols:
		distinctSymbols += [posTags[0][0]]
	if posTags[1][1] not in distinctSymbols:
		distinctSymbols += [posTags[1][1]]
	if '!!!!!@@@@@@' not in distinctSymbols:
		distinctSymbols += ['!!!!!@@@@@@']

fp.close()
fp = open("what","r")
for row in fp:
	row = row.lower()
	row = row.split(',')[-1]
	posTags = nltk.pos_tag(row.lower().strip().split())
	secondWord = posTags[1][0]
	if 'what' in row:
		secondWord = '!!!!!@@@@@@'
	whatPattern += [[posTags[0][0], secondWord, posTags[1][1]]]
	if posTags[0][0] not in distinctSymbols:
		distinctSymbols += [posTags[0][0]]
	if posTags[1][1] not in distinctSymbols:
		distinctSymbols += [posTags[1][1]]
	if secondWord not in distinctSymbols:
		distinctSymbols += [secondWord]
fp.close()
fp = open("when","r")
for row in fp:
	row = row.lower()
	row = row.split(',')[-1]
	posTags = nltk.pos_tag(row.lower().strip().split())
	secondWord = posTags[1][0]
	if 'when' in row:
		secondWord = '!!!!!@@@@@@'
	whenPattern += [[posTags[0][0], secondWord, posTags[1][1]]]
	if posTags[0][0] not in distinctSymbols:
		distinctSymbols += [posTags[0][0]]
	if posTags[1][1] not in distinctSymbols:
		distinctSymbols += [posTags[1][1]]
	if secondWord not in distinctSymbols:
		distinctSymbols += [secondWord]
fp.close()

# words to numbers
tagsDict = {}
i=0
for tag in distinctSymbols:
	tagsDict[tag] = i
	i+=1

classDict = {0:'what',1:'when',2:'affirmation'}

trainingX = []
trainingY = []
for patterns in whatPattern:
	trainingX += [[tagsDict[pattern] for pattern in patterns]]
	trainingY += [0]

for patterns in whenPattern:
	trainingX += [[tagsDict[pattern] for pattern in patterns]]
	trainingY += [1]

for patterns in affirmationPattern:
	trainingX += [[tagsDict[pattern] for pattern in patterns]]
	trainingY += [2]

trainingX = np.array(trainingX)
trainingY = np.array(trainingY)

#print whatPattern
#print whenPattern
#print affirmationPattern
#print tagsDict
#print trainingX,trainingY

# Training SVM
trainedModel = OneVsRestClassifier(sklearn.svm.SVC(random_state=0)).fit(trainingX,trainingY)

# Testing
fp = open("test.csv","r")
fp1 = open("testOut.csv","w")
writer = csv.writer(fp1,delimiter=",")
writer.writerow(['Question','Catagory'])
for row in fp:
	originalRow = row[:]
	row = row.lower().strip().split(",")[-1]
	posTags = nltk.pos_tag(row.split())
	catagory = ""
	try:
		secondWord = posTags[1][0]
		try:
			secondWordNumber = tagsDict[secondWord]
		except:
			secondWordNumber = tagsDict['!!!!!@@@@@@']
		#print [tagsDict[posTags[0][0]],secondWordNumber,tagsDict[posTags[1][1]]]
		testData = np.array([[tagsDict[posTags[0][0]],secondWordNumber,tagsDict[posTags[1][1]]]])
		catagory = classDict[trainedModel.predict(testData)[0]]
		if catagory=='affirmation':
			if 'what' in row:
				catagory = "what"
			elif 'when' in row:
				catagory = "when"
	except Exception as e:
		catagory = "Unknown"
	#print [originalRow.strip(),catagory]
	#input('next?')
	writer.writerow([originalRow.strip(),catagory])
fp.close()
fp1.close()
