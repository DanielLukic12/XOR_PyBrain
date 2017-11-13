# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:19:04 2017

@author: Daniel Lukic
"""

def plt_err(trainerList, learningRates, trainEpochs):
    """This Function Plots the Training Errors and Test Errors

    Args:
        trainerList: List which contains modules with the parameters 
        that gave the minimal validation error.
        
        learningRates: List of Learning Rates.
        trainEpochs: List of Convergence Epochs.

    Returns:
        A Figure wiht Training Errors and Test Errosrs for the Training

    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(15, 8))
                        
    for i in np.arange(np.size(trainerList)):
        
        trainer = trainerList[i]
        training_error = trainer.trainingErrors
        validation_errors = trainer.validationErrors
    
        x_train_err = np.arange(0, len(training_error))
        x_val_err = np.arange(0, len(validation_errors))
        
        plt.subplot(2,2, i+1)
        plt.semilogx(x_train_err, training_error, 'r', label='MSE Train')
        plt.xlabel('epochs', fontsize=18)
        plt.ylabel('MSE', fontsize=18)
        plt.semilogx(x_val_err, validation_errors, 'b', label='MSE Test')
        plt.grid()
        plt.legend()
        plt.title('MSE for lr = ' + str(learningRates[i]) + ' epoch_conv = ' 
        + str(trainEpochs[i]), fontsize=20)
        
    plt.tight_layout()  
    plt.show()
        