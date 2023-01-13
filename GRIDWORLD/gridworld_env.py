## Script: gridworld_env.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from pyglet.gl import *

import gym
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A simple gridworld environment for OpenAI Gym.
    
    The gridworld has a size of NxM, and the agent starts in the top-left corner.
    The agent can move up, down, left, or right, and the actions are deterministic.
    The agent gets a reward of 1 for reaching the goal state (bottom-right corner),
    and a reward of -1 for entering a cell with a "hole" (indicated by -1 in the grid).
    The episode ends when the agent reaches the goal or falls in a hole.
    
    The observation space is a 2D NumPy array of size NxM, representing the grid.
    The action space is a discrete space with 4 actions (up, down, left, right).
    """
    def __init__(self, N=5, M=5):
        self.N = N
        self.M = M
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(N, M), dtype=np.float32)
        self.grid = np.zeros((N, M))
        self.grid[-1, -1] = 1 # goal state
        #self.grid[-2, -2] = -1 # hole
        self.grid[-4:-1, -4:-1] = -1 # big hole

        self.pos = (0, 0)

        self.action_dict = dict()
        self.action_dict[0] = 'UP'
        self.action_dict[1] = 'DOWN'
        self.action_dict[2] = 'LEFT'
        self.action_dict[3] = 'RIGHT'

    def step(self, action):

        ret_obs = [self.grid,self.pos]
        ret_rew = [0,0]

        i, j = self.pos
        
        if action == 0:  # up
            i -= 1
        elif action == 1:  # down
            i += 1
        elif action == 2:  # left
            j -= 1
        elif action == 3:  # right
            j += 1
        
        # check if the agent has reached the edge of the grid
        if i < 0 or i >= self.N or j < 0 or j >= self.M:
            return ret_obs, [-1,0], False, {}
        
        # check if the agent has fallen in a hole
        if self.grid[i, j] == -1:
            self.pos = (i, j)
            return ret_obs, [0,-1], False, {}
        
        self.pos = (i, j)
        ret_obs = (self.grid,self.pos)

        if i == self.N-1 and j == self.M-1:  # goal state
            return ret_obs, [1,0], True, {}
        
        return ret_obs, [0,0], False, {}

    def reset(self):
        self.pos = (0, 0)
        return self.grid

    def render(self, mode='human'):
        self.render_gridworld(self.grid, self.pos)

    def render_gridworld_before(self, grid, pos,i):
        """
        Render the gridworld and save the image.
        
        Args:
            grid: a 2D NumPy array representing the grid
            pos: a tuple (i, j) representing the current position of the agent
        """
        plt.clf() 
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.imshow(grid, interpolation='nearest', cmap='gray')
        ax1.plot(pos[1], pos[0], 'ro', markersize=10)
        
        ax2.text(0.1, 0.5, 'Timestep '+str(i), horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_frame_on(False)
        plt.savefig('gridworld.png')
        plt.close(fig)

    def render_gridworld_after(self, grid, pos,i,action,reward):
        """
        Render the gridworld and save the image.
        
        Args:
            grid: a 2D NumPy array representing the grid
            pos: a tuple (i, j) representing the current position of the agent
        """
        plt.clf() 
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        ax1.imshow(grid, interpolation='nearest', cmap='gray')
        ax1.plot(pos[1], pos[0], 'ro', markersize=10)
        
        ax2.text(0.1, 0.5, 'Timestep '+str(i)+'\nTook action '+str(self.action_dict[action])+'\nGot Reward '+str(reward), horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_frame_on(False)
        plt.savefig('gridworld.png')
        plt.close(fig)


    def close(self):
        pass # you can implement this method if you need to close any resources


