# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 19:14:43 2019

@author: rdoce
"""

import numpy as np
import math

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
        
        self.Weight1Gradients=np.dot(np.multiply(np.transpose(np.dot(np.transpose(np.multiply(dErr_dOut,dOut_dNet2)),self.SecondWeights)),dHidden_dNet1),np.transpose(self.Input))
        self.Update()
        
    def Update(self):
        self.FirstWeights=self.FirstWeights - self.learningRate*self.Weight1Gradients
        self.SecondWeights=self.SecondWeights - self.learningRate*self.Weight2Gradients
        
    def Train(self,TrainingData,Labels,ValidationProp,Iterations):
        self.bestFirstWeights = []
        self.bestSecondWeights = []
        self.propCorrect = [] # proportion Correct during validation
        nTrain=math.floor(len(TrainingData)*(1-ValidationProp))
        nValidation=len(TrainingData)-nTrain
        for i in range(Iterations):
            print(i)
            nCorrect=0 # Counter for number of correct classifications
            for j in range(nTrain):
                inputTrain=TrainingData[j]
                inputTrain=inputTrain.reshape(len(inputTrain),1)
                labelTrain = Labels[:,j]
                labelTrain=labelTrain.reshape(len(labelTrain),1)
                self.ForwardPass(inputTrain)
                self.ComputeError(labelTrain)
                self.Backprop(labelTrain)
            for k in range(nValidation):
                inputValidation = TrainingData[nTrain+k] # input for validation
                inputValidation = inputValidation.reshape(len(inputValidation),1)
                labelValidation = Labels[:,nTrain+k] # label for validation instance
                labelValidation = labelValidation.reshape(len(labelValidation),1)
                maxVal=-1 # This variable is used to determine which output is predicted: maximum value in output layer corresponds to prediction
                self.ForwardPass(inputValidation)
                for l in range(self.nOut):
                    if self.Output[l]>maxVal:
                        maxVal = self.Output[l]
                        maxPos = l # position of max value in output
                    if labelValidation[l]==1:
                        corrPos = l # correct output position
                if maxPos==corrPos:
                    nCorrect=nCorrect+1
            self.propCorrect.append(nCorrect/nValidation)
            if max(self.propCorrect) == nCorrect/nValidation:
                self.bestFirstWeights=self.FirstWeights
                self.bestSecondWeights=self.SecondWeights
                
            
                
                ### Here : check if the output is correct (highest value = prediction)
                # Then, compute the accuracy overall
                # Add a variable that tracks the accuracy through the iterations

def VectoriseLabels(Labels):
    # This function takes a set of labels, where each label is a single number 
    # (encoding position) and returns them in vector format
    
    nLabels = len(Labels)
    nCategories = max(Labels)
    newLabels = np.zeros([nCategories,nLabels])
    Labels=Labels-1
    newLabels[Labels,range(nLabels)]=1
    return newLabels
            
            
        
        
    
        
    
        
        