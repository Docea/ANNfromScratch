#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:46:33 2019

@author: rdocea
"""

import numpy as np
import math

class LayerwiseNetwork:
    def __init__(self):
        self.Config=[]
        self.Structure=[]
        
        
    def InputLayer(self,*argv):
        config = ['Input']
        for arg in argv:
            config.append(arg)
        self.Config.append(config)
                
            
    def ConvolutionLayer(self,dimA,dimB,padding):
        config = ['Convolution',dimA,dimB,padding]
        self.Config.append(config)        
        
            
    def DenseLayer(self,*argv):
        config = ['Dense']
        for arg in argv:
            config.append(arg)
        self.Config.append(config)
        
        
    def Activation(self,activationType):
        # E.g.: 'Sigma'
        config = ['Activation',activationType]
        self.Config.append(config)
        
    def Maxpool(self,size,stride):
        config = ['Maxpool',size,stride]
        self.Config.append(config)
        
    def Compose(self):
        pointer=0
        
        while pointer < len(self.Config):
            if self.Config[pointer][0]=='Input':
                layerStructure = ['Input']
                if len(self.Config[pointer][1:])>1:
                    layerStructure.append(np.zeros(self.Config[pointer][1:]))
                else:
                    layerStructure.append(np.zeros([self.Config[pointer][1],1]))
                self.Structure.append(layerStructure)
                inDims=[] # Dimension of input to structure
                outDims=self.Config[pointer][1:] # Dimension of output from structure 
            
            if self.Config[pointer][0]=='Convolution':
                inDims = outDims
                layerStructure = ['Convolve']
                if self.Config[pointer][3]=='Valid':
                    outDims[:] = [x-self.Config[pointer][1]+1 for x in outDims]
                    layerStructure.append(np.zeros(outDims))
                if self.Config[pointer][3]=='Same':
                    layerStructure.append(np.zeros(outDims))
                layerStructure.append(np.random.normal(0,1,[self.Config[pointer][1],self.Config[pointer][2]]))
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Dense':
                inDims = outDims
                layerStructure=['Dense']
                if len(outDims)>1:
                    n = 1
                    for i in range(len(outDims)):
                        n=n*outDims[i]
                else:
                    n = outDims
                layerStructure.append(np.zeros([self.Config[pointer][1],1]))
                layerStructure.append(np.random.normal(0,1,[n,self.Config[pointer][1]]))
                outDims=[self.Config[pointer][1],1]
                layerStructure.append(np.random.normal(0,1,outDims))
                self.Structure.append(layerStructure)
            
            if self.Config[pointer][0]=='Activation':
                inDims=outDims
                layerStructure=['Activation']
                layerStructure.append(np.zeros(outDims))
                layerStructure.append(self.Config[pointer][1])
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Maxpool':
                inDims=outDims
                outDims[:] = [math.floor(x/self.Config[pointer][1]) for x in outDims]
                layerStructure = ['Maxpool']
                layerStructure.append(np.zeros(outDims))
                layerStructure.append(self.Config[pointer][1:3])
                self.Structure.append(layerStructure)
                
            if pointer == (len(self.Config)-1):
                self.Structure.append(list(self.Structure[pointer]))                
            
            pointer += 1
        
        self.Structure[pointer][0] = 'Output'
        
        pointer = len(self.Structure)-1
        while pointer >= 0:
            
            layerStructure = self.Structure[pointer]
                
            if layerStructure[0]=='Activation':
                layerStructure.append(np.zeros(np.shape(layerStructure[1])))
            
            if layerStructure[0]=='Output':
                layerStructure.append(np.zeros(np.shape(layerStructure[1])))
                
            if layerStructure[0]=='Dense':
                layerStructure.append(np.zeros(np.shape(layerStructure[2])))
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1])))
                
            if layerStructure[0]=='Maxpool':
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1])))
                
            if layerStructure[0]=='Convolve':
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer][2])))
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1])))
                
            
            
            self.Structure[pointer]=layerStructure
            
            pointer -= 1
                
            

        

    def Forwardpass(self,Input):
        pointer = 0
        while pointer < len(self.Structure):
            if self.Structure[pointer][0]=='Input':
                self.Structure[pointer][1]=Input
            
            if self.Structure[pointer][0]=='Convolve':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=len(self.Structure[pointer][2])
                
                for i in range(nRows-filterSize+1):
                    for j in range(nCols-filterSize+1):
                        self.Structure[pointer][1][i,j]=np.sum(np.multiply(self.Structure[pointer][2],self.Structure[pointer-1][1][i:i+filterSize,j:j+filterSize]))
            
            if self.Structure[pointer][0]=='Maxpool':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=len(self.Structure[pointer][2])
                
                for i in range(math.floor(nRows/filterSize)):
                    for j in range(math.floor(nCols/filterSize)):
                        self.Structure[pointer][1][i,j]=np.max(self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize])
            
            if self.Structure[pointer][0]=='Activation':
                if self.Structure[pointer][2]=='Sigmoid':
                    self.Structure[pointer][1]=1/(1+np.exp(-self.Structure[pointer-1][1]))
                    
            if self.Structure[pointer][0]=='Dense':
                shape = len(np.shape(self.Structure[pointer-1][1]))
                if shape>1: #Check if flat
                    self.Structure[pointer-1][1]=np.ndarray.flatten(self.Structure[pointer-1][1])
                self.Structure[pointer][1]=np.reshape(np.dot(np.transpose(self.Structure[pointer][2]),self.Structure[pointer-1][1]), np.shape(self.Structure[pointer][1])) + self.Structure[pointer][3] # Multiply input to hidden layer by weights
                    
            if self.Structure[pointer][0]=='Output':
                self.Structure[pointer][1]=self.Structure[pointer-1][1]
            pointer += 1
            
    def ComputeError(self,Label):
        self.Output = self.Structure[len(self.Structure)-1][1]
        self.Error=np.sum((self.Output-Label)**2)
        
    def GetOutput(self):
        self.Output = self.Structure[len(self.Structure)-1][1]
        return self.Output
        
    def Backpropagate(self,Output,Label):
        pointer = len(self.Structure)-1
        while pointer >= 0:
            
            if self.Structure[pointer][0]=='Output':
                self.Structure[pointer][len(self.Structure[pointer])-1]=self.Structure[pointer][1]-np.reshape(Label,np.shape(self.Structure[pointer][1]))
                
            if self.Structure[pointer][0]=='Activation':
                if self.Structure[pointer][2]=='Sigmoid':
                    # Calculate Schar Product between next layer's output and activation derivative
                    self.Structure[pointer][len(self.Structure[pointer])-1]=np.multiply(Recast(np.multiply(self.Structure[pointer][1],(1-self.Structure[pointer][1]))),self.Structure[pointer+1][len(self.Structure[pointer+1])-1])
                    
            if self.Structure[pointer][0]=='Dense':
                self.Structure[pointer][-2]=np.dot(Recast(self.Structure[pointer-1][1]),np.transpose(self.Structure[pointer+1][-1])) # Calculates the derivatives for the weights 
                self.Structure[pointer][-1]=np.dot(self.Structure[pointer][2],self.Structure[pointer+1][-1])
                
            if self.Structure[pointer][0]=='Maxpool':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=len(self.Structure[pointer][2])
                
                self.Structure[pointer+1][-1]=np.reshape(self.Structure[pointer+1][-1],self.Structure[pointer][2])
                
                for i in range(math.floor(nRows/filterSize)):
                    for j in range(math.floor(nCols/filterSize)):
                        localMax=np.max(self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize])
                        # The below line looks complicated, but it is 3 copies of the same term (almost). It sets all but the maximum value currently under the maxpool filter window to 0
                        self.Structure[pointer][-1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize]  =  (self.Structure[pointer+1][-1][i,j])  *  (self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize]==localMax)
                        
            if self.Structure[pointer][0]=='Convolve':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=len(self.Structure[pointer][2])
                
                for i in range(nRows-filterSize+1): # Change this for variable stride
                    for j in range(nCols-filterSize+1): # Change this for variable stride
                        self.Structure[pointer][-1][i:i+filterSize,j:j+filterSize] += self.Structure[pointer+1][-1][i,j]*self.Structure[pointer][2] # This calculates the derivative with respect to the input to the layer
                        self.Structure[pointer][-2] += self.Structure[pointer-1][-1][i:i+filterSize,j:j+filterSize]*self.Structure[pointer+1][-1][i,j] # Calculates derivative with respect to filter weights

                        
            pointer -= 1
                
        ##### Before: add biases to Compose() and Forwardpass()
        ##### Add extra place at end of each piece in self.Structure for a placeholder for backpropagation
        
        
    def Update(self):  
        pointer = 0
        while pointer < len(self.Structure):
            
            if self.Structure[pointer][0]=='Dense':
                self.Structure[pointer][2]=self.Structure[pointer][2]-self.Structure[pointer][-2]
                                        
            if self.Structure[pointer][0]=='Convolve':
                self.Structure[pointer][2]=self.Structure[pointer][2]-self.Structure[pointer][-2]
            
            pointer += 1
            
            
    
            
def Recast(input):
    if len(np.shape(input))==1:
        input=np.reshape(input,[len(input),1])
    return input
    
            
            
                        
    
            
                
            
                
            
        
        