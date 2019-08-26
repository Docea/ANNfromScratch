#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:47:37 2019

@author: rdocea
"""

# Test Script for Layerwise Network
from matplotlib import pyplot as plt
import numpy as np
import LayerwiseNetwork as layerNet

Net = layerNet.LayerwiseNetwork()
Net.InputLayer(100,100)
Net.ConvolutionLayer(6,6,'Valid')
Net.ConvolutionLayer(4,4,'Valid')   
Net.Maxpool(2,2)
Net.Activation('Sigmoid')
Net.DenseLayer(15)
Net.Activation('Sigmoid')
Net.DenseLayer(3)

Net.Compose()

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
    


