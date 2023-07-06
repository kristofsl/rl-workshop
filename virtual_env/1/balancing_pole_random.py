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

class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def act(self, state: np.array) -> int:
        return self.env.action_space.sample()

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

def record_random_agent():
    # trying the random agent first
    env = gym.make('CartPole-v1',render_mode = 'rgb_array_list')
    
    # wrap the environment for recording
    wrapped_env = RecordVideo(env = env, video_folder = './random-video', episode_trigger = lambda x: x % 25 == True)

    # fetch the observation dimensions from the environment (4 dimensions)
    observation_space_size = wrapped_env.observation_space.shape[0]
    
    # create a random agent
    random_agent = RandomAgent(wrapped_env)
    
    # evaluate the random agent
    rewards, steps = evaluate(random_agent, wrapped_env, 50, observation_space_size)
        
    # evaluation results
    median_steps = np.median(steps)
    mean_steps = np.mean(steps)
    std_steps = np.std(steps)
        
    print(f'median steps = {median_steps}')
    print(f'std steps    = {std_steps}')
    print(f'mean steps   = {mean_steps}')    
        

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

# record a random agent    
record_random_agent()

exit()