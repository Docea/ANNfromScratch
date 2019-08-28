#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:47:37 2019

@author: rdocea
"""

# Test Script for Layerwise Network
import sys
sys.path.insert(1,'../')
from matplotlib import pyplot as plt
import numpy as np
import LayerwiseNetwork as layerNet
import numpy as np
import pandas as pd
import math


Net = layerNet.LayerwiseNetwork()
Net.InputLayer(28,28)
Net.ConvolutionLayer(6,6,'Valid')
Net.ConvolutionLayer(4,4,'Valid')   
Net.Maxpool(2,2)
Net.Activation('Sigmoid')
Net.DenseLayer(18)
Net.Activation('Sigmoid')
Net.DenseLayer(9)
Net.Compose()


trainData = pd.read_csv('train.csv',',')
trainData = trainData.values
trainLabels = layerNet.VectoriseLabels(trainData[:,0]) # Assigns each label a separate output
trainData = trainData[:,1:] # Separates training data from labels
trainData = (trainData/255)-0.5 # Mean-normalising data and rescaling data
trainData = np.transpose(trainData)
trainData = np.reshape(trainData,[28,28,42000])
trainData = list(np.transpose(trainData))
trainData = trainData[0:1000]
testData = pd.read_csv('test.csv',',') # Importing test data, although *CURRENTLY UNUSED*
testData = testData.values
testData = np.transpose(testData)

trainData = np.random.rand(6,28,28)
trainData = list(trainData)

Net.Train(trainData,trainLabels,0.2,1000,0.1)


'''
inputSize = len(trainData[:,1]) # Determines the size of the input layer to the neural network
hiddenSize = 27 # Sets the size of the hidden layer (hidden layer size << input layer size)
outputSize = len(trainLabels) # Sets output layer size in accordance with number of labers
learningRate = 0.1 # Sets learning rate
nIterations = 50 # Sets number of iterations of learning on the training set
validationProp = 0.2 # The proportion of the training data that is assigned for validation


b = Net.Structure

testInput=np.random.rand(100,100)
errorRecord = []
for i in range(1000):
    Net.Forwardpass(testInput)
    Net.GetOutput()
    Net.ComputeError([1,1,1])
    Net.Backpropagate(Net.Output,[1,1,1])
    Net.Update()
    errorRecord.append(Net.Error)
    


'''