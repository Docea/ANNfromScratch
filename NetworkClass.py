# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:14:43 2019

@author: rdoce
"""

import numpy as np

class Network:
    def __init__(self, nIn, nHid, nOut,activationFunc,learningRate):
        self.nIn = nIn
        self.nHid = nHid
        self.nOut = nOut
        self.activationFunc=activationFunc
        self.learningRate = learningRate
        
        self.FirstWeights = np.random.normal(0,1,[nHid,nIn])
        self.SecondWeights = np.random.normal(0,1,[nOut,nHid])

        self.FirstBiases = np.random.normal(0,1,[nHid,1])
        self.SecondBiases = np.random.normal(0,1,[nOut,1])
        
        #self.Output = np.zeros(nOut,1)
        
        
    def ForwardPass(self,Input):
        self.Input=Input
        self.netHidden=np.dot(self.FirstWeights,Input)+self.FirstBiases
        self.Hidden=self.Activation(self.netHidden)
        self.netOutput=np.dot(self.SecondWeights,self.Hidden)+self.SecondBiases
        self.Output=self.Activation(self.netOutput)
    
    def ComputeError(self,Label):
        self.Error=np.sum((self.Output-Label)**2)
     
           
    def Activation(self,Layer):
        if self.activationFunc=='Sigmoid':
            return 1/(1+np.exp(-Layer))
        
    def Backprop(self,Label):
        dErr_dOut=(self.Output-Label)
        if self.activationFunc=='Sigmoid':
            dOut_dNet2=self.Output*(1-self.Output)
        dNet2_dW2=self.Hidden
        
        self.Weight2Gradients=np.dot(np.multiply(dErr_dOut,dOut_dNet2),np.transpose(dNet2_dW2))
        
        dNet2_dHidden=self.SecondWeights
        if self.activationFunc=='Sigmoid':
            dHidden_dNet1=self.Hidden*(1-self.Hidden)
        dNet1_dW1=self.Input
        
        self.Weight1Gradients=np.dot(np.multiply(np.transpose(np.dot(np.transpose(np.multiply(dErr_dOut,dOut_dNet2)),self.SecondWeights)),self.Hidden),np.transpose(self.Input))
        self.Update
        
    def Update(self):
        self.FirstWeights=self.FirstWeights - self.learningRate*self.Weight1Gradients
        self.SecondWeights=self.SecondWeights - self.learningRate*self.Weight2Gradients
    
    def Train(self,TrainingData,Labels,ValidationProp,Iterations):
        self.propCorrect = [] 
        nTrain=floor(len(TrainingData[:][1])*(1-ValidationProp))
        nValidation=len(TrainingData[:][1])-nTrain
        for i in range(Iterations):
            for j in range(nTrain):
                inputTrain=TrainingData[j][:]
                labelTrain = Labels[:][j]
                self.ForwardPass(inputTrain)
                self.ComputeError(labelTrain)
                self.Backprop(labelTrain)
            for k in range(nValidation):
                inputValidation = TrainingData[nTrain+k][:] # input for validation
                labelValidation = Labels[:][nTrain+k] # label for validation instance
                nCorrect=0 # Counter for number of correct classifications
                maxVal=-1 # This variable is used to determine which output is predicted: maximum value in output layer corresponds to prediction
                self.ForwardPass(inputValidation)
                for l in range(self.nOut):
                    if self.Output[l]>maxVal:
                        maxVal = self.Output[l]
                        maxPos = l # position of max value in output
                    if labelTrain[l]==1:
                        corrPos = l # correct output position
                if maxPos==corrPos:
                    nCorrect=nCorrect+1
                self.propCorrect.append(nCorrect/nValidation)
    
                    
                
                ### Here : check if the output is correct (highest value = prediction)
                # Then, compute the accuracy overall
                # Add a variable that tracks the accuracy through the iterations

            
            
        
        
    
        
    
        
        