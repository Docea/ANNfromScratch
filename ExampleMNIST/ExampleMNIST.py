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
#Net.ConvolutionLayer(6,'Valid')       # Uncomment to include layer
#Net.ConvolutionLayer(4,'Valid')       # Uncomment to include layer
Net.Maxpool(2,2)                      # Uncomment to include layer
Net.DenseLayer(20)
Net.Activation('Sigmoid')
Net.DenseLayer(10)
Net.Activation('Sigmoid')
Net.Compose()


trainData = pd.read_csv('train.csv',',')
trainData = trainData.values
trainLabels = layerNet.VectoriseLabels(trainData[:,0]) # Assigns each label a separate output
trainLabels = list(np.transpose(trainLabels))
trainData = trainData[:,1:] # Separates training data from labels
trainData = (trainData/255)-0.5 # Mean-normalising data and rescaling data
trainData = np.reshape(trainData,[42000,28,28])
trainData = list(trainData)

# testData = pd.read_csv('test.csv',',') # test data *CURRENTLY UNUSED*
# testData = testData.values
# testData = np.transpose(testData)

validationFrequency = 1000
Net.Train(trainData,trainLabels,0.05,10,0.1,validationFrequency)
# validationFrequency indicates with what frequency (in samples) validation should be performed

# Plots accuracy on the training set (for n = validationFrequency samples) and accuracy on the validation set
nValids = len(Net.trainCorrect) # number of validations
nSamples = np.arange(1*validationFrequency,(nValids+1)*validationFrequency,validationFrequency)
plt.plot(nSamples,Net.trainCorrect,label='Train')
plt.plot(nSamples,Net.validCorrect,label='Validation')
plt.show()
plt.xlabel('No. Samples Trained')
plt.ylabel('Accuracy')
plt.title('Accuracy VS Samples Trained (Epoch Size: 40000)')
plt.legend()