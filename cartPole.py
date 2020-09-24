#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:55:20 2020

@author: alikemalcelenk
"""

import gym
import numpy as np
from collections import deque
from keras.models import Sequention #ANN
from keras.layers import Dense #ANN
from keras.optimizers import Adam #ANN
import random

class DQLAgent:
    
    def __init__(self, env):
       #hyperparameter - parameter
       pass
   
    def buildModel(self):
        # neural network for deep q-learning
        pass
    
    def remember(self, state, action, reward, nextState, done):
        #storage
        pass
    
    def act(self, state):
        #acting
        pass
    
    def replay(self, batchSize):
        #training
        pass
    
    def adaptiveEGreedy(self):
        pass
        


if __name__ == '__main__':
    
    #initialize env and agent
    
    episodes = 100
    for e in range(episodes):
        
        #initialize environment
    
        while True:
            
            # act
            
            # step  -  done buradan gelcek
            
            # remember
            
            # update state
            
            # replay
            
            # adjust epsilon
            
            if done:
                break