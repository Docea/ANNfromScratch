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
testData = pd.read_csv('test.csv',',')
testData = testData.values

sampleInput = trainData[0][1:]
sampleInput = sampleInput.reshape(len(sampleInput),1)
sampleLabel = trainData[0][0]

inputSize = len(sampleInput)
hiddenSize = round(inputSize*1.25)
outputSize = 1
learningRate=0.1

#####################

Net = net.Network(inputSize,hiddenSize,1,'Sigmoid',0.1)

#Input = np.array([[0],[1],[0]])
Input = (sampleInput-255/2)/255
#Label = np.array([[0.2],[0.6],[0.4]])
Label = sampleLabel
Net.ForwardPass(Input)
Net.ComputeError(Label)
Net.Backprop(Label)

Weights1=Net.FirstWeights
Weights2=Net.SecondWeights
Biases1=Net.FirstBiases
Biases2=Net.SecondBiases

Error = Net.Error
Output = Net.Output
WeightGrads1 = Net.Weight1Gradients
WeightGrads2 = Net.Weight2Gradients

op = []
op2 = []
op3 = []

for i in range(10000):
    Net.ForwardPass(Input)
    Net.ComputeError(Label)
    Net.Backprop(Label)

    Net.Update()
    
    newWeights = Net.FirstWeights
    Output = Net.Output
    
    op.append(Output[0][0])
    #op2.append(Output[1][0])
    #op3.append(Output[2][0])
    
plt.plot(op)
plt.plot(op2)
plt.plot(op3)
plt.show()