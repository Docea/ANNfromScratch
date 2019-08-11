#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:30:20 2019

@author: rdocea
"""

#MAIN SCRIPT#

import NetworkClass as net
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#trainData = pd.read_csv('train.csv',',')
#trainData = trainData.values
#testData = pd.read_csv('test.csv',',')
#testData = testData.values

sampleInput = trainData[0][1:]
sampleInput = sampleInput.reshape(len(sampleInput),1)
sampleLabel = trainData[0][0]

inputSize = len(sampleInput)
hiddenSize = round(inputSize*1.25)
outputSize = 1
learningRate=0.1

Net = net.Network(inputSize,hiddenSize,outputSize,'Sigmoid',learningRate)

#Input = np.random.rand(5,1)
Input = sampleInput
Label = sampleLabel
Label = 1
#Label = np.random.rand(2,1)
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

Net.Update()

op = []
for i in range(500):
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
    
    op.append(Output[0][0])
    
    Net.Update()
    
print("max weight grads: ", np.max(WeightGrads1))
print("min weight grads: ", np.min(WeightGrads1))
plt.plot(op)