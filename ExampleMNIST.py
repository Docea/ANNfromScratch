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

################ Uncomment for Extracting Data from CSVs ######################
import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv('train.csv',',')
trainData = trainData.values
#trainData = trainData[41000:,:]
trainLabels = net.VectoriseLabels(trainData[:,0])
trainData = trainData[:,1:]
trainData = (trainData/255)-0.5
trainData = np.transpose(trainData)
testData = pd.read_csv('test.csv',',')
testData = testData.values
testData = np.transpose(testData)

inputSize = len(trainData[:,1])
hiddenSize = round(inputSize*1.25)
outputSize = len(trainLabels)
learningRate = 0.1
nIterations = 25

#####################
# Initialise network structure

Net = net.Network(inputSize,hiddenSize,outputSize,'Sigmoid',learningRate)

# Train network, providing training data and corresponding labels
Net.Train(trainData,trainLabels,0.2,nIterations)

plt.plot(Net.propCorrect)
plt.xlabel("Iterations")
plt.ylabel("Accuracy on Validation Set")
title = "Learning Rate = {}".format(learningRate)
plt.title(title)
plt.show()