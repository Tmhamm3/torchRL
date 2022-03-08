import torch
import numpy as np
from Agent.DQN import DQN
from Agent.PPO_Discrete import PPO_Discrete
from Agent.PPO_Continuous import PPO_Continuous
from Agent.Dagger import DAGGER
from Environment.Gym import Gym
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer
import configparser, os
from pathlib import Path

class Trainer():
    def __init__(self, configuration_parameters):
        super(Trainer, self).__init__()
        self.configuration_parameters = configuration_parameters
        self.agent_type = configuration_parameters['TrainingAgent']['type']
        self.set_object_parameters()
        self.initialize_run()
        self.steps = 0
        self.episodes = 0
        self.train_or_evaluate()

    def train(self):
        while self.continue_session():
            self.environment.display()
            state = self.environment.get_state()
            action = self.training_agent.get_action(state)
            next_state,reward,end_of_episode,world_info = self.environment.step(action)
            self.training_agent.memory_buffer.store(action,state,reward,next_state,end_of_episode)
            self.steps += 1

            if world_info == 'Threshold met':
                print(f'Performance Threshold has been met in {self.episodes} episodes')
                break

            if world_info == 'Adjust Learning Rate':
                self.training_agent.adjust_learning_rate(self.learning_rate_adjustments[self.learning_rate_idx])
                self.learning_rate_idx += 1

            if end_of_episode:
                self.episodes += 1
                self.training_agent.episodes_completed += 1
                if 'PPO' in self.agent_type:
                    self.training_agent.update()

            if not 'PPO' in self.agent_type:
                self.training_agent.update()

        if self.save_agent:
            save_file = os.path.join(self.save_agent_folder,self.save_agent_file_name)
            self.training_agent.save(file_name=save_file)
            temp = os.getcwd() + self.save_agent_folder + '/' + self.save_agent_file_name
        self.environment.close()

    def evaluate(self):
        while self.continue_session():
            self.environment.display()
            state = self.environment.get_state()
            action = self.training_agent.get_action(state,evaluate=True)
            next_state,reward,end_of_episode,world_info = self.environment.step(action)
            self.episode_reward += reward
            self.steps += 1

            if end_of_episode:
                self.rewards[self.episodes] = self.episode_reward
                self.episode_reward = 0
                self.episodes += 1

        self.environment.close()

    def train_or_evaluate(self):
        assert hasattr(self,'session_goal') and self.session_goal in ['train','evaluate'], 'The purpose of this run is non clear. Check configuration file for session_goal attribute'
        assert hasattr(self,'session_tracker') and self.session_tracker in ['episodic', 'step_wise'], "Missing or incorrect training_tracker attribute, check configuration file for attribute entry"
        if self.session_tracker == 'episodic':
            assert hasattr(self,'max_episodes'), "Missing max_episodes attribute, check configuration file for attribute entry"
        else:
            assert hasattr(self,'max_steps'), "Missing max_steps attribute, check configuration file for attribute entry"
        if self.session_goal == 'train':
            self.train()
        else:
            self.evaluate()

    def continue_session(self):
        if self.session_tracker == 'episodic':
            return self.episodes < self.max_episodes
        else:
            return self.steps < self.max_steps

    def initialize_run(self):
        self.make_env()
        self.make_agent()

    def make_agent(self):
        agent_class = globals()[self.agent_type]
        self.training_agent = agent_class(self.configuration_parameters,self.environment.summary_writer)
        if self.load_agent:
            self.training_agent.load(self.load_agent_file_name)

    def make_buffer(self):
        self.memory_buffer = SAMemoryBuffer(self.configuration_parameters)

    def make_env(self):
        self.environment = Gym(self.configuration_parameters)
        self.configuration_parameters['Environment']['action_space_dim'] = self.environment.action_space_dim
        self.configuration_parameters['Environment']['action_space_type'] = self.environment.action_space_type
        self.configuration_parameters['Environment']['state_space_dim'] = self.environment.state_space_dim
        self.configuration_parameters['Environment']['reward_space_dim'] = self.environment.reward_space_dim

    def set_object_parameters(self):
        for key, value in self.configuration_parameters['Trainer'].items():
            setattr(self,key,value)
        if hasattr(self,'learning_rate_adjustments'):
            assert 'lr_thresholds' in self.configuration_parameters['Environment'].keys(), 'Missing environment learning rate threshold values'
            assert len(self.learning_rate_adjustments) == len(self.configuration_parameters['Environment']['lr_thresholds']), 'Learning rate adjustment parameters have different lengths'
            self.learning_rate_idx = 0
        for key in self.configuration_parameters.keys():
            self.configuration_parameters[key]['seed'] = self.seed