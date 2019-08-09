# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:14:43 2019

@author: rdoce
"""

import numpy as np

class Network:
    def __init__(self, nIn, nHid, nOut,activationFunc=str(),learningRate):
        self.activationFunc=activationFunc
        
        self.FirstWeights = np.random.rand(nHid,nIn)
        self.SecondWeights = np.random.rand(nOut,nHid)

        self.FirstBiases = np.random.rand(nHid,1)
        self.SecondBiases = np.random.rand(nOut,1)
        
        self.learningRate = learningRate
        #self.Output = np.zeros(nOut,1)
        
        
    def ForwardPass(self,Input):
        self.netHidden=np.dot(self.FirstWeights,Input)+self.FirstBiases
        self.Hidden=self.Activation(self.netHidden)
        self.netOutput=np.dot(Hidden,self.SecondWeights)+self.SecondBiases
        self.Output=self.Activation(self.netOutput)
    
    def ComputeError(self,Label):
        self.Error=np.sum((self.Output-Label)**2)
     
           
    def Activation(self,Layer):
        if self.activationFunc=='Sigmoid':
            return 1/(1+np.exp(-Layer))
        
    def Backprop(self,Label):
        dErr_dOut=(self.Output-Label)
        dOut_dNet2=np.exp(self.netOutput)+2+np.exp(-self.netOutput)
        dNet2_dW2=self.Hidden
        
        Weight2Gradients=np.dot(np.dot(np.transpose(dErr_dOut),dOut_dNet2),np.transpose(dNet2_dW2))

        
        
    
        
        
    
        
    
        
        