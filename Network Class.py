# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:14:43 2019

@author: rdoce
"""

import numpy as np

class Network:
    def __init__(self, nIn, nHid, nOut,activationFunc):
        self.activationFunc=activationFunc
        
        self.FirstWeights = np.random.rand(nHid,nIn)
        self.SecondWeights = np.random.rand(nOut,nHid)

        self.FirstBiases = np.random.rand(nHid,1)
        self.SecondBiases = np.random.rand(nOut,1)
        
        #self.Output = np.zeros(nOut,1)
        
        
    def ForwardPass(self,Input):
        Hidden=np.dot(self.FirstWeights,Input)+self.FirstBiases
        Hidden=self.ActivationFunc(Hidden)
        self.Output=np.dot(Hidden,self.SecondWeights)+self.SecondBiases
        #self.Output=self.ActivationFunc(self.Output)
    
    def ComputeError(self,Label):
        self.Error=np.sum((self.Output-Label)^2)
        
    def ActivationFunc(self,Layer):
        if self.activationFunc == 'Sigmoid'
            return 1/(1+np.exp(-Layer))
        
        
        
    
        
        