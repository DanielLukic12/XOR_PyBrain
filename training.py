# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:19:04 2017

@author: Daniel Lukic
"""

import pybrain as pyb
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


def train_nn(learningRate, maxEpochs, dataTrain):
    """This builds and trains the Neural Net for given Data

    Args:
        learningRate: Learning Rate for Neural Net.
        maxEpochs: Maximal Epochs to train.
        dataTrain: Tuple of Training and Testing Data

    Returns:
        Returns the build neural net and the module with the parameters 
        that gave the minimal validation error.

    """

    # Call Training and validation data
    dataInput = dataTrain[0]
    dataTarget = dataTrain[1]
    
    # Define pybrain structure for training and validation data
    DS = SupervisedDataSet(2, 1)
    
    # Add training and validation Data to pybrain structure 
    for i in range(len(dataInput)):
        DS.addSample(dataInput[i, :], dataTarget[i])
        
    # Build neural net and define parameters
    net = buildNetwork(2, 2, 1, bias = True, hiddenclass =  pyb.TanhLayer)
        
    # Define learning algorithm
    trainer = BackpropTrainer(net, DS, learningrate = learningRate, momentum = 0.99)
    # Start training of neural net
    trainer.trainUntilConvergence(maxEpochs=maxEpochs)
   
    return net, trainer