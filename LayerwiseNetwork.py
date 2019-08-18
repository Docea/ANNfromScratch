#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:46:33 2019

@author: rdocea
"""

class LayerwiseNetwork:
    def __init__(self):
        self.Structure=[]
        
        
    def InputLayer(self,*argv):
        config = ['Input']
        for arg in argv:
            config.append(arg)
        self.Structure.append(config)
                
            
    def ConvolutionLayer(self,dimA,dimB,padding):
        config = ['Convolution',dimA,dimB,padding]
        self.Structure.append(config)        
        
            
    def HiddenLayer(self,*argv):
        config = ['Hidden']
        for arg in argv:
            config.append(arg)
        self.Structure.append(config)
        
        
    def OutputLayer(self,*argv):
        config = ['Output',argv]
        
    def Activation(self,activationType):
        # E.g.: 'Sigma'
        config = [activationType]
        
    def Maxpool(self):
        config = ['Maxpool']
        
    def Compose(self):
        