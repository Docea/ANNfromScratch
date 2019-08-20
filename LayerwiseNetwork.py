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
        
            
    def HiddenLayer(self,*argv):
        config = ['Hidden']
        for arg in argv:
            config.append(arg)
        self.Config.append(config)
        
        
    def OutputLayer(self,*argv):
        config = ['Output']
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
                layerStructure.append(np.random.rand(self.Config[pointer][1],self.Config[pointer][2]))
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Hidden':
                inDims = outDims
                layerStructure=['Hidden']
                if len(outDims)>1:
                    n = 1
                    for i in range(len(outDims)):
                        n=n*outDims[i]
                else:
                    n = outDims
                layerStructure.append(np.zeros([self.Config[pointer][1],1]))
                layerStructure.append(np.random.rand(n,self.Config[pointer][1]))
                outDims=[self.Config[pointer][1],1]
                self.Structure.append(layerStructure)

                
            if self.Config[pointer][0]=='Output':
                inDims = outDims
                outDims = self.Config[pointer][1]
                if isinstance(inDims, list):
                    inDims=max(inDims)
                if isinstance(outDims, list):
                    outDims=max(outDims)
                layerStructure=['Output']
                layerStructure.append(np.zeros([self.Config[pointer][1],1]))
                layerStructure.append(np.random.rand(inDims,outDims))
                outDims=[self.Config[pointer][1],1]
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
            
            pointer += 1

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
                    
            if self.Structure[pointer][0]=='Hidden':
                shape = len(np.shape(self.Structure[pointer-1][1]))
                if shape>1: #Check if flat
                    self.Structure[pointer-1][1]=np.ndarray.flatten(self.Structure[pointer-1][1])
                self.Structure[pointer][1]=np.dot(np.transpose(self.Structure[pointer][2]),self.Structure[pointer-1][1]) # Multiply input to hidden layer by weights
                    
            if self.Structure[pointer][0]=='Output':
                self.Structure[pointer][1]=np.dot(np.transpose(self.Structure[pointer][2]),self.Structure[pointer-1][1]) # Multiply input to hidden layer by weights
            
            pointer += 1
            
    def ComputeError(self,Label):
        self.Output = self.Structure[len(self.Structure)][1]
        self.Error=np.sum((self.Output-Label)**2)
        
    def getOutput(self):
        self.Output = self.Structure[len(self.Structure)][1]
        return self.Output
        
    #def Backpropagate(self,Output,Label):
        
            
            
            
                        
                
            
                
            
                
            
        
        