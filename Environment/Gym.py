from abc import ABC
import gym
from Environment.Environment import Environment
import torch
import torch.nn as nn
import torch.optim as optim


class Gym(Environment):
    def __init__(self, environment_parameters):
        super(Gym,self).__init__(environment_parameters)
        self.environment = gym.make(self.environment_name)
        self.environment.seed(self.seed)
        self.set_action_space()
        self.set_state_space()
        self.display()
        self.state = self.environment.reset()
        self.reward_space_dim = 1


    def step(self,action):
        if self.action_space_type =='Discrete':
            self.action = action.argmax().item()
        else:
            self.action = action

        self.state, self.step_reward, self.end_of_episode, self.world_info = self.environment.step(self.action)
        self.episode_reward += self.step_reward
        self.steps_total+=1
        self.episode_length+=1

        if self.end_of_episode:
            print(f'Episode {self.episodes_completed} Reward: {self.episode_reward}')
            if self.threshold:
                self.world_info = self.check_episode_threshold()
            self.episodes_completed+=1
            self.summary_writer.add_scalar('episode_reward', self.episode_reward, self.episodes_completed)
            self.summary_writer.add_scalar('episode_length', self.episode_length, self.episodes_completed)
            self.episode_reward = 0
            self.episode_length = 0
            self.state = self.environment.reset()


        return torch.from_numpy(self.state.astype(float)), torch.tensor([self.step_reward]), torch.tensor([self.end_of_episode]), self.world_info

    def set_action_space(self):
        if type(self.environment.action_space) == gym.spaces.discrete.Discrete:
            self.action_space_type = 'Discrete'
            self.action_space_dim = self.environment.action_space.n
        else:
            self.action_space_type = 'Continuous'
            self.action_space_dim = 1
            for i in range(len(self.environment.action_space.shape)):
                self.action_space_dim *= self.environment.action_space.shape[i]

    def set_state_space(self):
        self.state_space_type=''
        if type(self.environment.observation_space) == gym.spaces.discrete.Discrete:
            self.state_space_type = 'Discrete'
            self.state_space_dim = self.environment.observation_space.n
        else:
            self.state_space_type == 'Continuous'
            self.state_space_dim = 1
            for i in range(len(self.environment.observation_space.shape)):
                self.state_space_dim *= self.environment.observation_space.shape[i]

    def check_episode_threshold(self):
        self.threshold_tracker[self.threshold_index] = self.episode_reward
        self.threshold_index = (self.threshold_index +1) % self.eval_threshold_episodes
        if hasattr(self,'eval_threshold_episodes') and self.threshold_tracker.mean() >= self.evaluation_threshold:
            return 'Threshold met'
        elif hasattr(self,'lr_thresholds'):
            if self.lr_idx < self.lr_size and self.threshold_tracker[self.threshold_index - self.lr_threshold_episodes:self.threshold_index].mean() >= self.lr_thresholds[self.lr_idx]:
                self.lr_idx += 1
                return 'Adjust Learning Rate'
        else:
            return 'Keep Playing'

    def get_state(self):
        return torch.from_numpy(self.state.astype(float))

    def display(self):
        if self.render:
            self.environment.render()

    def close(self):
        self.environment.close()