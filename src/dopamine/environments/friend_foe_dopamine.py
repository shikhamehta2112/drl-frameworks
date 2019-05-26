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
from .friend_foe_backup import FriendFoeEnvironment #First change for another environmnent

class CartPoleEnv2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def __init__(self):
        self.hidden_env = FriendFoeEnvironment() #Second change for another environmnent
        self.hidden_env.reset()

        #Here we should also return the observation and action space that are expected

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(shape=(6, 5), low=0, high=255, dtype=np.float32) #Third change for another environmnent (the shape)
        self.state = self.hidden_env.current_game._board[0]
        self.seed()
        self.viewer = None
        self.state = self.hidden_env.current_game._board[0]
        self.step_counter=0

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        #prev_reward= self.hidden_env.current_game._episode_return
        print("We did... "+str(action))
        TS= self.hidden_env.step(action)
        #TS = self.hidden_env.process_timestep(TS)
        reward=TS[1]
        #reward= self.hidden_env.current_game._episode_return -prev_reward

        self.state = self.hidden_env.current_game._board[0]

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
            print("Hidden reward..."+str(self.hidden_env._get_hidden_reward()))
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.hidden_env.reset()
        self.state = self.hidden_env.current_game._board[0]
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