#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:55:20 2020

@author: alikemalcelenk
"""
import sys
sys.path.append('/home/ec2-user/anaconda3/lib/python3.6/site-packages') #(package ları görmüyordu)->(EC2 Ubuntu)

import gym
import numpy as np
from collections import deque
from keras.models import Sequential #ANN kuracağımız için
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    
    def __init__(self, env):
       #hyperparameter - parameter
       self.stateSize = env.observation_space.shape[0] # input number for ANN
       self.actionSize = env.action_space.n
      
       self.gamma = 0.95
       self.learningRate = 0.001
       
       self.epsilon = 1
       self.epsilonDecay = 0.995
       self.epsilonMin = 0.01
       
       self.memory = deque(maxlen = 1000) # capacity = 1000
       
       self.model = self.buildModel()

    def buildModel(self):
        # neural network for deep q-learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.stateSize, activation = 'tanh')) #hidden layer
        model.add(Dense(self.actionSize, activation = 'linear')) #output layer
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learningRate))
        return model
    
    def remember(self, state, action, reward, nextState, done):
        #storage
        self.memory.append((state, action, reward, nextState, done))
    
    def act(self, state):
        #acting explore or exploit
        if random.uniform(0,1) <= self.epsilon: #explore
            return env.action_space.sample() #rastgele 2 action dan birini yapacak
        else: #exploit
            actValues = self.model.predict(state)
            return np.argmax(actValues[0])

        
    
    def replay(self, batchSize):
        #training
        if len(self.memory) < batchSize: # batchSize kadar memory topladıktan sonra training işlemi başlıyor
            return
        miniBatch = random.sample(self.memory, batchSize)
        for state, action, reward, nextState, done in miniBatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(nextState)[0])
                # amax([[1, 2], [3, 4]]) ->  1,2,3,4]
            trainTarget = self.model.predict(state)
            trainTarget[0][action] = target
            self.model.fit(state, trainTarget, verbose = 0)
    
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay  # E = E * 0.995


if __name__ == '__main__':
    
    #initialize env and agent
    env = gym.make('CartPole-v0')
    agent = DQLAgent(env)
    
    batchSize = 16 # storage dan alıp kullandığımız parametre sayısı
    episodes = 100
    for e in range(episodes):
        
        #initialize environment
        state = env.reset()
        #array([ 0.02240576,  0.0233702 , -0.02742878,  0.03661042])
        
        state = np.reshape(state,[1,4])
        #array([[ 0.02240576,  0.0233702 , -0.02742878,  0æ.03661042]])
    
        time = 0
        while True:
            
            # act
            action = agent.act(state)
            
            # step  -  done buradan gelcek
            nextState, reward, done, _ = env.step(action)
            nextState = np.reshape(nextState,[1,4])
            
            # remember(storage)
            agent.remember(state, action, reward, nextState, done)

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
            
# %% test
#import time
trainedModel = agent
#env = gym.wrappers.Monitor(env, "./gym-results", force=True)    -    #if u use jupyter, should enable this line for rendering in jupyter screen
state = env.reset()
state = np.reshape(state, [1,4])
time_t = 0
while True:
    env.render()
    action = trainedModel.act(state)
    nextState, reward, done, _ = env.step(action)
    nextState = np.reshape(nextState, [1,4])
    state = nextState
    time_t += 1
    print(time_t)
    #time.sleep(0.4)
    if done:
        break
print("Done")

# %% Jupyter rendering    -    #if u use jupyter, should enable this line for rendering in jupyter screen
# import io
# import base64
# from IPython.display import HTML

# video = io.open('./gym-results/openaigym.video.%s.video000000.mp4' % env.file_infix, 'r+b').read()
# encoded = base64.b64encode(video)
# HTML(data='''
#     <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
# .format(encoded.decode('ascii')))
