from typing import Tuple, List, Callable, Union, Optional, Dict
import numpy as np
import os
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import time
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import uuid

class DeepQAgent:
    
    def __init__(self, env, memory_size, gamma, exploration_decay, layer_size, batch_size, exploration_min, learning_rate, extra_intermediate_layers = 0, exploration_rate = 1.0):
        self.__env = env
        self.__state_size = 4                                           # we have an 4 numbers that represent our size and this will be the input for our neural net
        self.__action_size = 2                                          # we have 2 possible actions (push the card to the left or to the right)
        self.__memory = deque(maxlen=memory_size)                       # memory for storing our experiences
        self.__layer_size_1 = layer_size                                # nn layer 1 width
        self.__layer_size_2 = layer_size                                # nn layer 2 width
        self.__extra_intermediate_layers = extra_intermediate_layers    # extra intermediate layers?
        self.__learning_rate = learning_rate                            # learning rate
        self.__model = self._build_model()                              # our model
        self.__gamma = gamma                                            # discount rate
        self.__exploration_rate = exploration_rate                      # exploration rate
        self.__exploration_min = exploration_min                        # min value for exploration
        self.__exploration_decay = exploration_decay                    # decay in the exploration rate (moving from exploration to exploitation)
        self.__step_counter = 0                                         # keep track of the steps and update the target model after target_model_delay steps
        self.__sample_batch_size = batch_size                           # how much samples to we take for the learning step
        self.__uuid = str(uuid.uuid4())                                 # unique uid for the model
       

    def add_to_memory(self, state, action, reward, next_state, done):
        self.__memory.append((state, action, reward, next_state, done))

    def disable_exploration(self):
        self.__exploration_rate = 0
        self.__exploration_min = 0
        
    def get_agent_uuid(self):
        return self.__uuid   
        
    def save_model(self):
        self.__model.save(f'model-{self.__uuid}.keras')
            
    def replay(self):
        
        if len(self.__memory) <= self.__sample_batch_size:
            return
        else:
            mini_batch = random.sample(self.__memory, self.__sample_batch_size)

            current_state = np.zeros((self.__sample_batch_size, self.__state_size))
            next_state = np.zeros((self.__sample_batch_size, self.__state_size))
            target_q_values = np.zeros((self.__sample_batch_size, self.__state_size))

            action = np.zeros(self.__sample_batch_size, dtype=int)
            reward = np.zeros(self.__sample_batch_size)
            done = np.zeros(self.__sample_batch_size,dtype=bool)

            for i in range(self.__sample_batch_size):
                current_state[i] = mini_batch[i][0]   # state
                action[i] = mini_batch[i][1]          # action
                reward[i] = mini_batch[i][2]          # reward
                next_state[i] = mini_batch[i][3]      # next_state
                done[i] = mini_batch[i][4]            # done
                                         
            target = self.__model.predict(current_state,verbose=0)
            Qvalue_ns = self.__model.predict(next_state,verbose=0)

            for i in range(self.__sample_batch_size):
                if done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    target[i][action[i]] = reward[i] + self.__gamma * (np.amax(Qvalue_ns[i]))


            self.__model.fit(current_state, target, batch_size=self.__sample_batch_size, epochs=1, verbose=0)

            if self.__exploration_rate > self.__exploration_min:
                self.__exploration_rate *= self.__exploration_decay
            

    def act(self, state: np.array) -> int:
        if np.random.rand() < self.__exploration_rate:
            random_action = self.__env.action_space.sample()
            return random_action
        else:    
            actions = self.__model.predict(state, verbose=0)             # get a model prediction                   
            best_action = np.argmax(actions[0])                          # get the output index with the highest value
            return best_action

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.__layer_size_1, input_dim=self.__state_size, activation='relu'))
        model.add(Dense(self.__layer_size_2, activation='relu'))
        if (self.__extra_intermediate_layers > 0):
            for i in range(self.__extra_intermediate_layers + 1): 
                model.add(Dense(self.__layer_size_2, activation='relu'))
        model.add(Dense(self.__action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.__learning_rate))
        return model  