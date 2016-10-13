# QuestionClassifier

## Description
This library classifies questions given in test.csv into what, when, affirmation and unknown classes

It considers questions given in files what, when and affirmation as training data corresponding to each type. Initial set of examples are added. Further more examples can be added to these files if required

Output is written to file testOut.csv

## Dependencies
Python(2.7)

Python Packages: nltk, sklearn, numpy, csv

## Running Code
  1. Add training data to what, when and affirmation files as traning data. Some examples are already added sufficient to run code.
  2. Add the questions you want to calssify in test.csv file.
  3. Run code.py
  4. Check file testOut.csv for output

## Overview of Approach
  1. POS tagging is done using NLTK pos-tagger
  2. Using selected words and POS tags as features, multiclass SVM classifier is used for classification using one versus all approach and RBF kernal
