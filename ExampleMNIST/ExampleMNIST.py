#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:30:20 2019

@author: rdocea
"""

#MAIN SCRIPT#

import NetworkClass as net
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv('train.csv',',')
trainData = trainData.values
trainLabels = net.VectoriseLabels(trainData[:,0]) # Assigns each label a separate output
trainData = trainData[:,1:] # Separates training data from labels
trainData = (trainData/255)-0.5 # Mean-normalising data and rescaling data
trainData = np.transpose(trainData)
testData = pd.read_csv('test.csv',',') # Importing test data, although *CURRENTLY UNUSED*
testData = testData.values
testData = np.transpose(testData)

inputSize = len(trainData[:,1]) # Determines the size of the input layer to the neural network
hiddenSize = 27 # Sets the size of the hidden layer (hidden layer size << input layer size)
outputSize = len(trainLabels) # Sets output layer size in accordance with number of labers
learningRate = 0.1 # Sets learning rate
nIterations = 50 # Sets number of iterations of learning on the training set
validationProp = 0.2 # The proportion of the training data that is assigned for validation


# Initialise network structure
Net = net.Network(inputSize,hiddenSize,outputSize,'Sigmoid',learningRate)

# Train network, providing training data and corresponding labels
Net.Train(trainData,trainLabels,validationProp,nIterations)

# Plot results on validation set
plt.plot(Net.propCorrect)
plt.xlabel("Training Iterations")
plt.ylabel("Accuracy on Validation Set")
title = "Learning Rate = {}".format(learningRate)
plt.title(title)
plt.show()