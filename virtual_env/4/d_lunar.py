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
import optuna
from optuna.importance import get_param_importances
import uuid
import mlflow
from d_agent import *

UUID = str(uuid.uuid4())
ML_FLOW_EXPERIMENT_ID = mlflow.create_experiment("mlflow-rl-lunar-tracking-6-"+UUID)

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


def enhance_reward_signal(next_state, reward, done, truncated, steps):
    if truncated:
        return reward - 10
    elif done and (next_state[6] != 1 or next_state[7] != 1):
        return reward - 10
    else: 
        return reward

# random agent
class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def act(self, state: np.array) -> int:
        return self.env.action_space.sample()
    
def record_random_agent():
    # trying the random agent first
    env = gym.make('LunarLander-v2',
                   continuous = False,
                   gravity = -10.0,
                   enable_wind = False,
                   wind_power = 15.0,
                   turbulence_power = 1.5,
                   render_mode = 'rgb_array_list')
    
    # wrap the environment for recording
    wrapped_env = RecordVideo(env = env, video_folder = './random-video', episode_trigger = lambda x: x % 25 == True)

    # fetch the observation dimensions from the environment
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

    
def sample_hyper_parameters(trial: optuna.trial.Trial) -> Dict:

    # how many experiences do we use for one mini batch?
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # what is the size of the memory in the memory replay mechanism? (from which we sample our mini batches)
    memory_size = trial.suggest_int("memory_size", 500, 20000)
    
    # learning rate
    gamma = trial.suggest_float('gamma', 0.9, 0.99)

    # the decay of our exploration 
    exploration_decay = trial.suggest_categorical('exploration_decay', [0.9, 0.95, 0.98, 0.99])

    # size of the layers
    layer_size = trial.suggest_categorical('layer_size',[128, 256, 512])

    # a min value for exploration (during training)
    exploration_min = trial.suggest_float('exploration_min',0.001, 0.1)

    # how many episodes do we use for training?
    episodes = trial.suggest_int("episodes", 600, 5000)

    # the learning rate for the adam optimizer / back prop
    learning_rate = trial.suggest_categorical('learning_rate',[0.00001,0.0001,0.001])

    # extra layers that are identical to the middle layer?
    extra_intermediate_layers = trial.suggest_categorical('extra_layers',[0,2])

    # update the target model with the main model information
    update_model_steps = trial.suggest_categorical('update_model_steps',[10, 20, 30])

    return {
        'batch_size': batch_size,
        'memory_size': memory_size,
        'gamma': gamma,
        'exploration_decay' : exploration_decay,
        'layer_size' : layer_size,
        'exploration_min' : exploration_min,
        'episodes' : episodes,
        'learning_rate' : learning_rate,
        'extra_intermediate_layers' : extra_intermediate_layers,
        'update_model_steps' : update_model_steps
    }    
    
# train loop for the deep q agent
def optimize_train(agent, env, observation_space_size, n_episodes):
    
    early_stopping_counter = 0

    for i in range(0, n_episodes):
        # reset and reshape the state
        state = np.reshape(env.reset()[0], [1, observation_space_size])
        
        print(f'new train episode {i} on : {datetime.datetime.now()}')

        # reset the internal state (learning counter)
        agent.reset_episode()
        
        # initialize done (fail) and truncated (success)
        done = False
        truncated = False
        
         # execute the garbase collector
        gc.collect()

        # clear all the memory
        keras.backend.clear_session()

        # logging purpose counter
        steps = 0
        rewards = 0
        
        while not (done or truncated):
            
            steps += 1

            # take action based on the state
            action = agent.act(state)

            # observe the signal from the environment after the action
            next_state, reward, done, truncated, info = env.step(action)
          
            # enhance the signal and merge the truncated and done signals
            reward = enhance_reward_signal(next_state, reward, done, truncated, steps) 
            
            rewards += reward
            
            # reshape the state 
            next_state = np.reshape(next_state, [1, observation_space_size])

            if done or truncated:
                mlflow.log_metric(key="episode_steps", value=steps, step=i)
                mlflow.log_metric(key="episode_reward", value=rewards, step=i)
                mlflow.log_metric(key="x_coord", value=next_state[0][0], step=i)
                mlflow.log_metric(key="y_coord", value=next_state[0][1], step=i)
                mlflow.log_metric(key="x_velocity", value=next_state[0][2], step=i)
                mlflow.log_metric(key="y_velocity", value=next_state[0][3], step=i)
                mlflow.log_metric(key="angle", value=next_state[0][4], step=i)
                mlflow.log_metric(key="angle_velocity", value=next_state[0][5], step=i)
                mlflow.log_metric(key="touch_left", value=next_state[0][6], step=i)
                mlflow.log_metric(key="touch_right", value=next_state[0][7], step=i)
                print(f'ended training episode with done {done} - truncated {truncated} - reward {reward} - step count {steps} -  next_state {next_state}')
                
                # some early stopping logic
                if done and next_state[0][6] == 1 and next_state[0][7] == 1 and rewards > 100:
                    print(f'incrementing the early stopping counter {early_stopping_counter}')
                    early_stopping_counter += 1
                else:
                    print('resetting the early stopping counter')
                    early_stopping_counter = 0
                    
                if early_stopping_counter > 5:
                    return
    
            # add the observation to the memory
            agent.add_to_memory(state, action, reward, next_state, done)

            # replay observations from the memory and learn if needed
            agent.replay()

            # replace the current state 
            state = next_state
            
            
            

def objective(trial: optuna.trial.Trial) -> float:

    # create an environment
    env = gym.make('LunarLander-v2',
                   continuous = False,
                   gravity = -10.0,
                   enable_wind = False,
                   wind_power = 15.0,
                   turbulence_power = 1.5,
                   render_mode = 'rgb_array_list')
    
    # wrap the environment for recording
    train_folder = f'./video-train-6/{UUID}'
    eval_folder = f'./video-eval-6/{UUID}'
    wrapped_env_train = RecordVideo(env = env, video_folder = train_folder, episode_trigger = lambda x: x % 200 == True)
    wrapped_env_evaluate = RecordVideo(env = env, video_folder = eval_folder, episode_trigger = lambda x: x % 20 == True)
    
    # fetch properties from environment
    observation_space_size = wrapped_env_train.observation_space.shape[0]
    action_space_size = wrapped_env_train.action_space.n
    
    print(f'observation space size : {observation_space_size}')
    print(f'action space size      : {action_space_size}')

    # get the hyper parameter config
    args = sample_hyper_parameters(trial)
    
    print(f'optuna run with params started with params : {trial.params}')

    # create the deep agent
    deep_agent = DeepQAgent(
        env,
        gamma = args['gamma'],
        memory_size = args['memory_size'],
        exploration_decay = args['exploration_decay'],
        layer_size = args['layer_size'],
        batch_size = args['batch_size'],
        exploration_min = args['exploration_min'],
        learning_rate = args['learning_rate'],
        update_model_steps = args['update_model_steps'],
        extra_intermediate_layers = args['extra_intermediate_layers']
    )
    
    RUN_NAME = f"run_{deep_agent.get_agent_uuid()}"
    
    mlflow.start_run(experiment_id=ML_FLOW_EXPERIMENT_ID, run_name=RUN_NAME)
    print('train loop started')    
    
    # reset all
    env.reset()
    wrapped_env_train.reset()
    wrapped_env_evaluate.reset()

    # log the optuna parameters
    mlflow.log_param("memory_size", args['memory_size'])
    mlflow.log_param("gamma", args['gamma'])
    mlflow.log_param("exploration_decay", args['exploration_decay'])
    mlflow.log_param("layer_size", args['layer_size'])
    mlflow.log_param("batch_size", args['batch_size'])
    mlflow.log_param("exploration_min", args['exploration_min'])
    mlflow.log_param("episodes", args['episodes'])
    mlflow.log_param("learning_rate", args['learning_rate'])
    mlflow.log_param("extra_intermediate_layers", args['extra_intermediate_layers'])
    mlflow.log_param("update_model_steps", args['update_model_steps'])
    
    # log the agent uid
    mlflow.log_param("agent_uid", deep_agent.get_agent_uuid())
    
    # train loop
    optimize_train(deep_agent,wrapped_env_train, observation_space_size,n_episodes=args['episodes'])

    # disable exploitation and only rely on the model for decisions
    deep_agent.disable_exploration()

    print('evaluation started')
    rewards, steps = evaluate(deep_agent, wrapped_env_evaluate, 50, observation_space_size)
    
    # evaluation results
    median_steps = np.median(steps)
    mean_steps = np.mean(steps)
    std_steps = np.std(steps)
    
    median_reward = np.median(rewards)
    mean_reward = np.mean(rewards)
    std_reward= np.std(rewards)
    
    # Track metrics
    mlflow.log_metric("median_reward", median_reward)
    mlflow.log_metric("std_reward", std_reward)
    mlflow.log_metric("mean_reward", mean_reward)

    print(f'median reward = {median_reward}')
    print(f'std reward    = {std_reward}')
    print(f'mean reward   = {mean_reward}')
    
    mlflow.log_metric("median_steps", median_steps)
    mlflow.log_metric("std_steps", std_steps)
    mlflow.log_metric("mean_steps", mean_steps)

    print(f'median steps = {median_steps}')
    print(f'std steps    = {std_steps}')
    print(f'mean steps   = {mean_steps}')

    # model saved as 
    deep_agent.save_model()
    
    # clear all the memory
    keras.backend.clear_session()
    
    mlflow.end_run()

    return median_reward

# record a random agent    
record_random_agent()

# create an optimized model
study_name = "rl-study-lunar-6"  
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name,direction='maximize',load_if_exists=True)

study.optimize(objective, n_trials=10, gc_after_trial=True)

# print param importances
print(get_param_importances(study))

exit()