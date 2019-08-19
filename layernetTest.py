#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:47:37 2019

@author: rdocea
"""

# Test Script for Layerwise Network

import LayerwiseNetwork as layerNet

Net = layerNet.LayerwiseNetwork()
Net.InputLayer(10,10)
Net.ConvolutionLayer(5,5,'Valid')
Net.ConvolutionLayer(3,3,'Valid')   
Net.Maxpool(2,2)
Net.Activation('Sigmoid')
Net.HiddenLayer(15)
Net.Activation('Sigmoid')
Net.OutputLayer(3)

Net.Compose()
