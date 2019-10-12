"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from .absent_supervisor import AbsentSupervisorEnvironment #First change for another environmnent
import tensorflow as tf 
import copy

class CartPoleEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def __init__(self):
        self.hidden_env = AbsentSupervisorEnvironment() #Second change for another environmnent
        self.hidden_env.reset()
        self.counter=0
        self.train=True
        self.test=False
        self.train_num_episodes=0
        self.test_num_episodes=0
        self.iteration=0
        self.average_train=0
        #Here we should also return the observation and action space that are expected
        self.log=1000
        self.averages=[]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(shape=(6,8,1), low=0, high=255, dtype=np.float32) #Third change for another environmnent (the shape)
        self.state = (self.hidden_env.current_game._board[0]).reshape(6,8,1)
        self.seed()
        self.viewer = None
        self.state = (self.hidden_env.current_game._board[0]).reshape(6,8,1)
        self.step_counter=0
        self._summary_writer = tf.summary.FileWriter("/home/shmehta/drl-frameworks/RayRainbowLogBoatRace")


    def step(self, action):
        self.counter+=1
        if self.counter%self.log==0 and self.train:
          self.counter=1
          self.train=False
          self.test=True
          self.test_num_episodes=0
          self.average_train=np.mean(np.array(self.averages), axis=0)
          self.averages=[]
        if self.counter%self.log==0 and self.test:
          self.test=False
          self.counter=1
          summary = tf.Summary(value=[
          tf.Summary.Value(tag='Train/NumEpisodes',
                         simple_value=self.train_num_episodes),
          tf.Summary.Value(tag='Train/AverageHiddenReturns',
                         simple_value=self.average_train),
          tf.Summary.Value(tag='Eval/NumEpisodes',
                         simple_value=self.test_num_episodes),
          tf.Summary.Value(tag='Eval/AverageHiddenReturns',
                         simple_value=np.mean(np.array(self.averages), axis=0))
          ])
          self.averages=[]
          self._summary_writer.add_summary(summary, copy.deepcopy(self.iteration))
          self.iteration+=1
          self.train=True
          self.train_num_episodes=0
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #prev_reward= self.hidden_env.current_game._episode_return
        print("We did... "+str(action)+" Iteration.. "+str(self.iteration))
        TS= self.hidden_env.step(action)
        #TS = self.hidden_env.process_timestep(TS)
        reward=TS[1]
        #reward= self.hidden_env.current_game._episode_return -prev_reward

        self.state = (self.hidden_env.current_game._board[0]).reshape(6,8,1)

        self.step_counter+=1
        if self.step_counter>100:
            done=True
            if reward==None:
                reward=0
        else:
            done=self.hidden_env.current_game.game_over
            if reward==None:
                reward=0
        if done:
            #print("Hidden reward..."+str(self.hidden_env._get_hidden_reward()))
            self.averages.append(self.hidden_env._get_hidden_reward())
            if self.train:
              self.train_num_episodes+=1
            else:
              self.test_num_episodes+=1
        return np.array(self.state), reward, done, {"hidden_reward":self.hidden_env._get_hidden_reward()}

    def reset(self):
        self.hidden_env.reset()
        self.state = (self.hidden_env.current_game._board[0]).reshape(6,8,1)
        self.step_counter = 0
        return np.array(self.state)

    def render(self, mode='human'):

        #if self.viewer is None:

        #if self.state is None: return None

        # Edit the pole polygon vertex

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
