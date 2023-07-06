from typing import Tuple, List, Callable, Union, Optional, Dict
import numpy as np
import os
import pathlib
import imageio
from pathlib import Path
import pandas as pd
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
from gym.wrappers import RecordVideo
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
from agent import *
import gc
from keras.callbacks import Callback       

# an evaluation loop for an agent
def evaluate(
    agent,
    env,
    n_episodes: int,
    observation_space_size: int
) -> Tuple[List, List]:
    
    # keep track of the reward and steps per episode
    reward_per_episode = []
    steps_per_episode = []

    for i in range(0, n_episodes):
        
        # reset and initial state from environment needs reshape to np array (tuple instead of array with dimensions 1,4)
        state = np.reshape(env.reset()[0], [1, observation_space_size])
        
        # initialize the cumulative steps and the rewards for this episode
        rewards = 0
        steps = 0
        
        # done = environment ended badly / truncated = environment ended after 500 episodes
        done = False
        truncated = False
        
        while not (done or truncated):
            
            # determine action based on the state
            action = agent.act(state)
            
            # take the action and observe the reward, new state, etc
            new_state, reward, done, truncated, info = env.step(action)
            
            # enchange the reward signal
            reward = enhance_reward_signal(new_state, reward, done, truncated, steps)

            # add the data from this step to the episode data
            rewards += reward
            steps += 1
            
            # reshape the new state 
            state = np.reshape(new_state, [1, observation_space_size])

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return reward_per_episode, steps_per_episode    
    
# train loop for the deep q agent
def train(agent, env, observation_space_size, n_episodes):
    # print('starting traing of agent ...')

    for i in range(0, n_episodes):
        # reset and reshape the state
        state = np.reshape(env.reset()[0], [1, observation_space_size])
        
        print(f'new train episode {i} on : {datetime.datetime.now()}')
        
        # initialize done (fail) and truncated (success)
        done = False
        truncated = False
        
        steps = 0
        
        while not (done or truncated):
            
            steps += 1

            # take action based on the state
            action = agent.act(state)

            # observe the signal from the environment after the action
            next_state, reward, done, truncated, info = env.step(action)
            
            # enhance the signal and merge the truncated and done signals
            reward = enhance_reward_signal(next_state, reward, done, truncated, steps)

            if done or truncated:
                print(f'ended training episode with done {done} - truncated {truncated} - reward {reward} - step count {steps}')

            # reshape the state 
            next_state = np.reshape(next_state, [1, observation_space_size])

            # add the observation to the memory
            agent.add_to_memory(state, action, reward, next_state, done)

            # replay observations from the memory and learn if needed
            agent.replay()

            # replace the current state 
            state = next_state
            
def enhance_reward_signal(next_state, reward, done, truncated, steps):
    # enhanced reward signal to stimulate the agent to keep the pole in the middle area
    if done:
        reward = -100
    elif truncated:
        reward = 100
    elif next_state[0] > 0.2:
        reward = -5
    elif next_state[0] < -0.2:
        reward = -5
        
    return reward

# an evaluation loop for an agent
def evaluate(
    agent,
    env,
    n_episodes: int,
    observation_space_size: int
) -> Tuple[List, List]:
    
    # keep track of the reward and steps per episode
    reward_per_episode = []
    steps_per_episode = []

    for i in range(0, n_episodes):
        
        # reset and initial state from environment needs reshape to np array (tuple instead of array with dimensions 1,4)
        state = np.reshape(env.reset()[0], [1, observation_space_size])
        
        # initialize the cumulative steps and the rewards for this episode
        rewards = 0
        steps = 0
        
        # done = environment ended badly / truncated = environment ended after 500 episodes
        done = False
        truncated = False
        
        while not (done or truncated):
            
            # determine action based on the state
            action = agent.act(state)
            
            # take the action and observe the reward, new state, etc
            new_state, reward, done, truncated, info = env.step(action)
            
            # enchange the reward signal
            reward = enhance_reward_signal(new_state, reward, done, truncated, steps)

            # add the data from this step to the episode data
            rewards += reward
            steps += 1
            
            # reshape the new state 
            state = np.reshape(new_state, [1, observation_space_size])

        reward_per_episode.append(rewards)
        steps_per_episode.append(steps)

    return reward_per_episode, steps_per_episode    
    
# train loop for the deep q agent
def train(agent, env, observation_space_size, n_episodes):
    # print('starting traing of agent ...')

    for i in range(0, n_episodes):
        # reset and reshape the state
        state = np.reshape(env.reset()[0], [1, observation_space_size])
        
        print(f'new train episode {i} on : {datetime.datetime.now()}')
        
        # initialize done (fail) and truncated (success)
        done = False
        truncated = False
        
        steps = 0
        
        # execute the garbase collector
        gc.collect()

        # clear all the memory
        keras.backend.clear_session()
        
        while not (done or truncated):
            
            steps += 1

            # take action based on the state
            action = agent.act(state)

            # observe the signal from the environment after the action
            next_state, reward, done, truncated, info = env.step(action)
            
            # enhance the signal and merge the truncated and done signals
            reward = enhance_reward_signal(next_state, reward, done, truncated, steps)

            if done or truncated:
                print(f'ended training episode with done {done} - truncated {truncated} - reward {reward} - step count {steps}')

            # reshape the state 
            next_state = np.reshape(next_state, [1, observation_space_size])

            # add the observation to the memory
            agent.add_to_memory(state, action, reward, next_state, done)

            # replay observations from the memory and learn if needed
            agent.replay()

            # replace the current state 
            state = next_state
            
def enhance_reward_signal(next_state, reward, done, truncated, steps):
    # enhanced reward signal to stimulate the agent to keep the pole in the middle area
    if done:
        reward = -100
    elif truncated:
        reward = 100
    elif next_state[0] > 0.2:
        reward = -5
    elif next_state[0] < -0.2:
        reward = -5
        
    return reward

# create an environment
env = gym.make('CartPole-v1',render_mode = 'rgb_array_list')

# wrap the environment for recording
wrapped_env = RecordVideo(env = env, video_folder = './video', episode_trigger = lambda x: x % 50 == True)

# fetch properties from environment
observation_space_size = wrapped_env.observation_space.shape[0]
action_space_size = wrapped_env.action_space.n

print(f'observation space size : {observation_space_size}')
print(f'action space size      : {action_space_size}')

# TODO FILL IN 
p_memory_size =                       # value from 500 - 5000
p_gamma =                             # value from 0.9 - 0.99
p_batch_size =                        # value from either [16,32,64,128]
p_exploration_decay =                 # value from 0.9 - 0.99
p_layer_size =                        # value from either [16,32,64,128,256]
p_exploration_min =                   # value from 0.0001 - 0.2
p_episodes =                          # value from 150 - 500
p_learning_rate =                     # value from 0.0001 - 0.001
p_extra_intermediate_layers =         # value from 0 - 1
p_episodes =                          # value from 100 - 1000

deep_agent = DeepQAgent(
    env,
    memory_size = p_memory_size,
    gamma = p_gamma,
    exploration_decay = p_exploration_decay,
    layer_size = p_layer_size,
    batch_size = p_batch_size,
    exploration_min = p_exploration_min,
    learning_rate = p_learning_rate,
    extra_intermediate_layers = p_extra_intermediate_layers,
)

print('train loop started')    

# reset all
env.reset()
wrapped_env.reset()

# train loop
train(deep_agent,env, observation_space_size,n_episodes=p_episodes)

# disable exploitation and only rely on the model for decisions
deep_agent.disable_exploration()

print('evaluation started')
rewards, steps = evaluate(deep_agent, wrapped_env, 50, observation_space_size)

# evaluation results
median_steps = np.median(steps)
mean_steps = np.mean(steps)
std_steps = np.std(steps)

median_reward = np.median(rewards)
mean_reward = np.mean(rewards)
std_reward= np.std(rewards)

print(f'median reward = {median_reward}')
print(f'std reward    = {std_reward}')
print(f'mean reward   = {mean_reward}')

print(f'median steps = {median_steps}')
print(f'std steps    = {std_steps}')
print(f'mean steps   = {mean_steps}')

# model saved as 
deep_agent.save_model()

# clear all the memory
keras.backend.clear_session()

exit()