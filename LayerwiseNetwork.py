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
                layerStructure.append(np.random.rand(self.Config[pointer][1],self.Config[pointer][2]))
                if self.Config[pointer][3]=='Valid':
                    outDims[:] = [x-self.Config[pointer][1]+1 for x in outDims]
                    layerStructure.append(np.zeros(outDims))
                if self.Config[pointer][3]=='Same':
                    layerStructure.append(np.zeros(outDims))
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
                layerStructure.append(np.random.rand(n,self.Config[pointer][1]))
                layerStructure.append(np.zeros([self.Config[pointer][1],1]))
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
                layerStructure.append(np.random.rand(inDims,outDims))
                layerStructure.append(np.zeros([self.Config[pointer][1],1]))
                outDims=[self.Config[pointer][1],1]
                self.Structure.append(layerStructure)
            
            if self.Config[pointer][0]=='Activation':
                inDims=outDims
                layerStructure=['Activation']
                layerStructure.append(self.Config[pointer][1])
                layerStructure.append(np.zeros(outDims))
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Maxpool':
                inDims=outDims
                outDims[:] = [math.floor(x/self.Config[pointer][1]) for x in outDims]
                layerStructure = self.Config[pointer]
                layerStructure.append(np.zeros(outDims))
                self.Structure.append(layerStructure)
            
            pointer += 1

            
            
            
                        
                
            
                
            
                
            
        
        