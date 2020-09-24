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
        #acting explore or exploit
        pass
    
    def replay(self, batchSize):
        #training
        pass
    
    def adaptiveEGreedy(self):
        pass
        


if __name__ == '__main__':
    
    #initialize env and agent
    env = gym.make('CartPole-v0')
    agent = DQLAgent(env)
    
    batchSize = 16 # storage dan alıp kullandığımız parametre sayısı
    episodes = 100
    for e in range(episodes):
        
        #initialize environment
        state = env.reset()  # []
        #array([ 0.02240576,  0.0233702 , -0.02742878,  0.03661042])
        
        state = np.reshape(state,[1,4])  # [[]]
        #array([[ 0.02240576,  0.0233702 , -0.02742878,  0.03661042]])
    
        time = 0 
        while True:
            
            # act
            action = agent.act(state)
            
            # step  
            nextState, reward, done, _ = env.step(action)
            nextState = np.reshape(nextState,[1,4])
            
            # remember(storage)
            agent.remember(state, action, reward, nextState, done):

            # update state
            state = nextState
            
            # replay
            agent.replay(batchSize)
            
            # adjust epsilon
            agent.adaptiveEGreedy()
            
            time += 1
            
            if done:
                print('Episode {}, time: {}'.format(e,time))
                break