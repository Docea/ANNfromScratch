#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 12:30:20 2019

@author: rdocea
"""

#MAIN SCRIPT#

import NetworkClass as net
import numpy as np

Net = net.Network(2,3,2,'Sigmoid',0.1)

Input = np.random.rand(2,1)
Label = np.array([[1],[0]])
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

for i in range(1000):
    Net.ForwardPass(Input)
    Net.ComputeError(Label)
    Net.Backprop(Label)

    Net.Update()
    
    newWeights = Net.FirstWeights
    Output = Net.Output
    
