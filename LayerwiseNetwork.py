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
        self.Config=[] # Contains configuration of the network
        self.Structure=[] # Contains relevant structures for the network
        self.learningRate=0.1 # Default learning rate value
        
        
    def InputLayer(self,*argv): # Define input dimensions of the network in 'argv'
        config = ['Input']
        dimensions = []
        for arg in argv:
            config.append(arg) 
        self.Config.append(config)
                
            
    def ConvolutionLayer(self,dim,padding): # Convolutional layer with square filter for dimensions 'dim'
        config = ['Convolution',dim,padding] 
        self.Config.append(config)        
        
            
    def DenseLayer(self,nNeurons): # Fully connected layer (flattens input layer by default). Specify number of neurons in this layer.
        config = ['Dense']
        config.append(nNeurons)
        self.Config.append(config)
        
        
    def Activation(self,activationType): # Convolutional layer with square filter for dimensions 'dim'. Specify activation type and the rest will be handled.
        # E.g.: 'Sigma'
        config = ['Activation',activationType]
        self.Config.append(config)
        
    def Maxpool(self,size,stride): # Maxpool layer. Specify the size of the square filter and the stride to be used.
        config = ['Maxpool',size,stride]
        self.Config.append(config)
        
    def Compose(self): # This function uses the self.Config array to generate the network structure in self.Structure
        pointer=0
        
        while pointer < len(self.Config):
            
            if self.Config[pointer][0]=='Input':
                
                '''
                Layer structure as follows (by position): 
                    0 - 'Input' declaration of layer type
                    1 - Output from layer (or rather what was provided as input)
                ''' 
                
                layerStructure = ['Input'] # [0]
                if len(self.Config[pointer][1:])>1:
                    layerStructure.append(np.zeros(self.Config[pointer][1:])) # Initialising input array/matrix [1]
                else:
                    layerStructure.append(np.zeros([self.Config[pointer][1],1])) # Initialising input array/matrix [1]
                self.Structure.append(layerStructure)
                inDims=[] # Dimension of input to structure
                outDims=self.Config[pointer][1:] # Dimension of output from structure 
            
            if self.Config[pointer][0]=='Convolution':
                
                '''
                Layer structure as follows (by position): 
                    0 - 'Convolve' declaration of layer type
                    1 - Final output from layer
                    2 - Weights for convolutional filter
                    3 - Bias term to be added to output from filter
                ''' 
                
                inDims = outDims
                layerStructure = ['Convolve'] # [0]
                if self.Config[pointer][2]=='Valid':
                    outDims[:] = [x-self.Config[pointer][1]+1 for x in outDims]
                    layerStructure.append(np.zeros(outDims)) # Initialisation of output array for Valid Padding [1]
                if self.Config[pointer][2]=='Same':
                    layerStructure.append(np.zeros(outDims)) # Initialisation of output array for Same Padding [1]
                layerStructure.append(np.random.normal(0,1,[self.Config[pointer][1],self.Config[pointer][1]])) # Initialisation of weights for square filter [2]
                layerStructure.append(np.random.normal(0,1,np.shape(layerStructure[1]))) # Initialisation of bias matrix to add to output [3]
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Dense':
                '''
                Layer structure as follows (by position): 
                    0 - 'Dense' declaration of layer type
                    1 - Final output from layer
                    2 - Weight matrix
                    3 - Bias term to be added to output 
                ''' 
                
                inDims = outDims
                layerStructure=['Dense'] # [0]
                if len(outDims)>1:
                    n = 1
                    for i in range(len(outDims)):
                        n=n*outDims[i]
                else:
                    n = outDims
                layerStructure.append(np.zeros([self.Config[pointer][1],1])) # Final output from layer [1]
                layerStructure.append(np.random.normal(0,1,[n,self.Config[pointer][1]])) # Weight matrix [2]
                outDims=[self.Config[pointer][1],1]
                layerStructure.append(np.random.normal(0,1,outDims)) # Bias term [3]
                self.Structure.append(layerStructure)
            
            if self.Config[pointer][0]=='Activation':
                '''
                Element structure as follows (by position): 
                    0 - 'Activation' declaration of operation
                    1 - Final output from element
                    2 - Declaration of activation type, i.e.: 'Sigmoid'... (more to come)
                ''' 
                
                inDims=outDims
                layerStructure=['Activation'] # [0]
                layerStructure.append(np.zeros(outDims)) # Output from stage [1]
                layerStructure.append(self.Config[pointer][1]) # Activation type [2]
                self.Structure.append(layerStructure)
                
            if self.Config[pointer][0]=='Maxpool':
                '''
                Layer structure as follows (by position): 
                    0 - 'Maxpool' declaration of layer type
                    1 - Final output from layer
                    2 - Dimensions of kernel and stride to be used
                    3 - Record of output dimensions from layer (used in backpropagation)
                ''' 
                
                
                inDims=outDims
                outDims[:] = [math.floor(x/self.Config[pointer][1]) for x in outDims] 
                layerStructure = ['Maxpool'] # [0]
                layerStructure.append(np.zeros(outDims)) # Output from layer [1]
                layerStructure.append(self.Config[pointer][1:3]) # Dimensions of dernel and stride [2]
                layerStructure.append(np.shape(layerStructure[1])) # Creates a record of the output dimensions [3]
                self.Structure.append(layerStructure)
                
            if pointer == (len(self.Config)-1): # Creates an output layer for use in backpropagation at a layer stage
                self.Structure.append(list(self.Structure[pointer]))                
            
            pointer += 1
        
        self.Structure[pointer][0] = 'Output'
        self.nOut = len(self.Structure[pointer][1]) # records number of output neurons
        
        pointer = len(self.Structure)-1
        while pointer >= 0: # Now, based on the feedforward structure defined within the loop above, the structure is expanded to contain structures for backpropagation 
            
            layerStructure = self.Structure[pointer]
                
            if layerStructure[0]=='Activation':
                layerStructure.append(np.zeros(np.shape(layerStructure[1]))) # dNet [-1]
            
            if layerStructure[0]=='Output':
                layerStructure.append(np.zeros(np.shape(layerStructure[1]))) # dErr/dOut [-1]
                
            if layerStructure[0]=='Dense':
                layerStructure.append(np.zeros(np.shape(layerStructure[2]))) # dWeights [-2]
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1]))) # dActivation [-1] 
                
            if layerStructure[0]=='Maxpool':
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1]))) # dMaxpool [-1]
                
            if layerStructure[0]=='Convolve':
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer][2]))) # dWeights [-2]
                layerStructure.append(np.zeros(np.shape(self.Structure[pointer-1][1]))) # dOut [-1]
                
            self.Structure[pointer]=layerStructure
            
            pointer -= 1
                
            

        

    def Forwardpass(self,Input): # This function defines a forward pass through the network, taking an input
        pointer = 0
        while pointer < len(self.Structure):
            if self.Structure[pointer][0]=='Input':
                self.Structure[pointer][1]=Input # 'Output' of the input layer is the input to the forwardpass function
            
            if self.Structure[pointer][0]=='Convolve':
                nRows=len(self.Structure[pointer-1][1][:]) # number of rows of matrix inputted to layer
                nCols=len(self.Structure[pointer-1][1][1][:]) # number of cols of matrix inputted to layer
                filterSize=len(self.Structure[pointer][2]) # side-length of filter kernel to be used
                
                # ! AT THE MOMENT, CONVOLUTIONAL LAYERS ASSUME STRIDE LENGTH OF 1 !
                
                # Iterating over input
                for i in range(nRows-filterSize+1): 
                    for j in range(nCols-filterSize+1): 
                        self.Structure[pointer][1][i,j]=np.sum(np.multiply(self.Structure[pointer][2],self.Structure[pointer-1][1][i:i+filterSize,j:j+filterSize]))
                    #           ^Output                                             ^Kernel Weights                  ^Section of Input from previous layer
                
                self.Structure[pointer][1] += self.Structure[pointer][3] # Addition of bias to output
            
            if self.Structure[pointer][0]=='Maxpool':
                nRows=len(self.Structure[pointer][1][:]) # number of rows of matrix output from layer
                nCols=len(self.Structure[pointer][1][1][:]) # number of cols of matrix output from layer
                filterSize=self.Structure[pointer][2][0] # side-length of kernel window for maxpool
                
                # ! AT THE MOMENT, MAXPOOL LAYERS ASSUME STRIDE LENGTH EQUAL TO THE SQUARE KERNEL SIZE !
                
                for i in range(math.floor(nRows/filterSize)):
                    for j in range(math.floor(nCols/filterSize)):
                        self.Structure[pointer][1][i,j]=np.max(self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize])
            
            if self.Structure[pointer][0]=='Activation':
                if self.Structure[pointer][2]=='Sigmoid':
                    self.Structure[pointer][1]=1/(1+np.exp(-self.Structure[pointer-1][1]))
                    
            if self.Structure[pointer][0]=='Dense':
                tempArray = self.Structure[pointer-1][1]
                shape = len(np.shape(tempArray))
                if shape>1: #Check if flat, and if not, flatten
                    tempArray = np.ndarray.flatten(tempArray) 
                self.Structure[pointer][1]=np.reshape(np.dot(np.transpose(self.Structure[pointer][2]),tempArray), np.shape(self.Structure[pointer][1])) # Multiply input to hidden layer by weights
                self.Structure[pointer][1] += self.Structure[pointer][3] # Adding bias to output
                
            if self.Structure[pointer][0]=='Output':
                self.Structure[pointer][1]=self.Structure[pointer-1][1]
            pointer += 1
            
    def ComputeError(self,Label):
        self.Output = self.Structure[-1][1]
        self.Error=np.sum((self.Output-Label)**2)
        
    def GetOutput(self):
        self.Output = self.Structure[-1][1]
        return self.Output
        
    def Backpropagate(self,Output,Label):
        pointer = len(self.Structure)-1
        while pointer >= 0:
            
            if self.Structure[pointer][0]=='Output':
                self.Structure[pointer][-1]=self.Structure[pointer][1]-np.reshape(Label,np.shape(self.Structure[pointer][1]))
                
            if self.Structure[pointer][0]=='Activation':
                if self.Structure[pointer][2]=='Sigmoid':
                    # Calculate Schar Product between next layer's output and activation derivative
                    self.Structure[pointer][-1]=np.multiply(self.Structure[pointer][1],(1-self.Structure[pointer][1])) # dNet_dOut = Output(1-Output) ; where the output is the output from the activation function
                
                if len(np.shape(self.Structure[pointer][-1]))==1: # Removes errors introduced by 1-D array
                    self.Structure[pointer][-1] = Recast(self.Structure[pointer][-1])
                self.Structure[pointer][-1]=np.multiply(self.Structure[pointer][-1],self.Structure[pointer+1][-1]) # Aggregates backpropagated derivatives
                    
            if self.Structure[pointer][0]=='Dense':
                tempArray = Recast(np.ndarray.flatten(self.Structure[pointer-1][1])) # Accounts for a non-flat output in previous layer
                self.Structure[pointer][-2]=np.dot(tempArray,np.transpose(self.Structure[pointer+1][-1])) # Calculates the derivatives for the weights 
                self.Structure[pointer][-1]=np.reshape(np.dot(self.Structure[pointer][2],self.Structure[pointer+1][-1]),np.shape(self.Structure[pointer][-1]))
                
            if self.Structure[pointer][0]=='Maxpool':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=self.Structure[pointer][2][0]
                
                self.Structure[pointer+1][-1]=np.reshape(self.Structure[pointer+1][-1],self.Structure[pointer][3])
                
                for i in range(math.floor(nRows/filterSize)):
                    for j in range(math.floor(nCols/filterSize)):
                        localMax=np.max(self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize])
                        # The below line looks complicated, but it is 3 copies of the same term (almost). It sets all but the maximum value currently under the maxpool filter window to 0
                        self.Structure[pointer][-1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize]  =  (self.Structure[pointer+1][-1][i,j])  *  (self.Structure[pointer-1][1][i*filterSize:i*filterSize+filterSize,j*filterSize:j*filterSize+filterSize]==localMax)
                        
            if self.Structure[pointer][0]=='Convolve':
                nRows=len(self.Structure[pointer-1][1][:])
                nCols=len(self.Structure[pointer-1][1][1][:])
                filterSize=len(self.Structure[pointer][2])
                
                self.Structure[pointer][-1] = np.zeros(np.shape(self.Structure[pointer][-1]))
                self.Structure[pointer][-2] = np.zeros(np.shape(self.Structure[pointer][-2]))
                
                for i in range(nRows-filterSize+1): # Change this for variable stride
                    for j in range(nCols-filterSize+1): # Change this for variable stride        
                        self.Structure[pointer][-1][i:i+filterSize,j:j+filterSize] += self.Structure[pointer+1][-1][i,j]*self.Structure[pointer][2] # This calculates the derivative with respect to the input to the layer
                        self.Structure[pointer][-2] += self.Structure[pointer-1][-1][i:i+filterSize,j:j+filterSize]*self.Structure[pointer+1][-1][i,j] # Calculates derivative with respect to filter weights

                        
            pointer -= 1
        
        
    def Update(self):  
        pointer = 0
        while pointer < len(self.Structure):
            
            if self.Structure[pointer][0]=='Dense':
                self.Structure[pointer][2]=self.Structure[pointer][2]-self.learningRate*self.Structure[pointer][-2]
                                        
            if self.Structure[pointer][0]=='Convolve':
                self.Structure[pointer][2]=self.Structure[pointer][2]-self.learningRate*self.Structure[pointer][-2]
            
            pointer += 1
            
            
    def Train(self,TrainingData,Labels,ValidationProp,Iterations,learningRate,validationFrequency):
        self.learningRate=learningRate
        self.validCorrect = [] # proportion Correct during validation
        self.trainCorrect = []
        nTrain=math.floor(len(TrainingData)*(1-ValidationProp))
        nValidation=len(TrainingData)-nTrain
        self.validErr = []
        self.trainErr = []
        validErr = 0
        trainErr = 0
        
        batchSize = validationFrequency
        splits = 0
        
        for i in range(Iterations):
            nCorrect = 0 # Counter for number of correct classifications
            splits = 0
            for j in range(nTrain):
                print("Iteration: ", i, "Instance: ", j)
                inputTrain=TrainingData[j]
                labelTrain = Labels[j]
                labelTrain=labelTrain.reshape(len(labelTrain),1)
                self.Forwardpass(inputTrain)
                self.GetOutput()
                self.ComputeError(labelTrain)
                self.Backpropagate(self.Output,labelTrain)
                self.Update()
                
                maxLabel = max(labelTrain)
                maxOutput = max(self.Output)
                for l in range(self.nOut):
                    if self.Output[l]==maxOutput:
                        maxPos = l # position of max value in output
                    if labelTrain[l]==maxLabel:
                        corrPos = l # correct output position
                if maxPos==corrPos:
                    nCorrect += 1
                self.ComputeError(labelTrain)
                trainErr += self.Error
            
            
                if math.floor(j/batchSize) > splits :
                    splits += 1
                    self.trainErr.append(trainErr)
                    trainErr = 0
                    self.trainCorrect.append(nCorrect/batchSize)
                    nCorrect = 0
                    for k in range(nValidation):
                        print("Iteration: ", i, "Instance: ", nTrain+k)
                        inputValidation = TrainingData[nTrain+k] # input for validation
                        labelValidation = Labels[nTrain+k] # label for validation instance
                        labelValidation = labelValidation.reshape(len(labelValidation),1)
                        self.Forwardpass(inputValidation)
                        self.GetOutput()
                        maxLabel = max(labelValidation)
                        maxOutput = max(self.Output)
                        for l in range(self.nOut):
                            if self.Output[l]==maxOutput:
                                maxPos = l # position of max value in output
                            if labelValidation[l]==maxLabel:
                                corrPos = l # correct output position
                        if maxPos==corrPos:
                            nCorrect += 1
                        self.ComputeError(labelValidation)
                        validErr += self.Error
                    self.validCorrect.append(nCorrect/nValidation)
                    self.validErr.append(validErr)
                    validErr = 0
                    nCorrect = 0
            
def VectoriseLabels(Labels):
    # This function takes a set of labels, where each label is a single number 
    # (encoding position) and returns them in vector format
    
    nLabels = len(Labels)
    nCategories = max(Labels+1)
    newLabels = np.zeros([nCategories,nLabels])
    Labels=Labels-1
    newLabels[Labels,range(nLabels)]=1
    return newLabels


            
def Recast(input):
    if len(np.shape(input))==1:
        input=np.reshape(input,[len(input),1])
    return input
    
            
            
                        
    
            
                
            
                
            
        
        