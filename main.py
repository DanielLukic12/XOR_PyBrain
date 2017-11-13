# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:18:19 2017

@author: Daniel Lukic
"""

import sys
sys.path.append('/home/daniel/Dropbox/UNI - Kopie/Bach/xor')
import training as train
import plt_error as plt_err
import numpy as np

# Define maximal epochs for training
maxEpochs = 500
# Define learning rates for training
learningRates = [0.01, 0.005 ,0.001, 0.0005]

# Input Data for Training and Evaluation 
dataInput = np.transpose([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1] , 
                          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
# Target Data for Training and Evaluation 
dataTarget = np.transpose([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
dataTrain = [dataInput, dataTarget]
trainerList = []
trainEpochs = []

# Train neural net for each learning rate
for learningrate in learningRates:
    
    # Start training of neural net
    net, trainer = train.train_nn(learningrate, maxEpochs, dataTrain)
    
    # Test the trained neural net and print the 
    print ''
    print '--------------------------------------------------'
    print 'Test of NN:', 'by learning rate: ', learningrate
    print 'x| y| out'
    print '---------'
    print '1| 0|', net.activate([1, 0])
    print '1| 1|', net.activate([1, 1])
    print '0| 1|', net.activate([0, 1])
    print '0| 0|', net.activate([0, 0])
    print 'Epoch to Convergence', trainer.epoch
    print '--------------------------------------------------'
    print ''
    
    trainerList.append(trainer)
    trainEpochs.append(trainer.epoch)

# Plot Errors from Training
plt_err.plt_err(trainerList, learningRates, trainEpochs)