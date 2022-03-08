import os
from copy import copy
import torch
from torch.distributions import Categorical
from Agent.Agent import Agent
from Agent.DQN import DQN
from Agent.PPO import PPO
from Utilities.Utils import NetworkBuilder

class DAGGER(Agent):
    def __init__(self,configuration_parameters,summary_writer):
        super(DAGGER,self).__init__(configuration_parameters,summary_writer)
        self.set_input_output_dim()
        self.set_save_attributes()
        self.load_expert_agent()
        self.make_buffer()
        self.actor = self.network(self.configuration_parameters['Actor'])
        self.optimizer = getattr(torch.optim,self.optimizer,None)(params=self.parameters(),lr=self.learning_rate)
        self.update = self.behavior_cloning
        self.get_expert_action = True

    def behavior_cloning(self):
        minibatch = self.memory_buffer.get_minibatch()
        actions, states, rewards, next_states, end_of_episodes = self.unpack_minibatch(minibatch)

        imitator_action_probabilities= self.actor(states).float()
        if self.configuration_parameters['TrainingAgent']['loss_function'] == 'MSELoss':
            actions = actions.detach().float()
        else:
            actions = actions.argmax(dim=-1).detach()

        if len(imitator_action_probabilities.size()) > len(actions.size()):
            imitator_action_probabilities = imitator_action_probabilities.squeeze(dim=-1)

        self.optimizer.zero_grad()
        self.loss = self.loss_function(imitator_action_probabilities,actions)
        self.loss.backward()
        self.optimizer.step()


        if self.episodes_completed >= self.expert_episodes:
            self.update = self.dagger
            self.get_expert_action = False
            self.memory_buffer.reset_buffer()

    def dagger(self):
        minibatch = self.memory_buffer.get_minibatch()
        actions, states, rewards, next_states, end_of_episodes = self.unpack_minibatch(minibatch)
        states = states.detach()
        next_states = next_states.detach()
        imitator_action_probabilities = self.actor(states).float()
        if self.configuration_parameters['TrainingAgent']['loss_function'] == 'MSELoss':
            expert_actions = self.expert_agent.get_action(states,evaluate=True).detach().float()
        else:
            expert_actions = self.expert_agent.get_action(states,evaluate=True).argmax(dim=-1).detach()

        self.optimizer.zero_grad()
        if len(imitator_action_probabilities.size()) > len(expert_actions.size()):
            imitator_action_probabilities = imitator_action_probabilities.squeeze(dim=-1)

        self.loss = self.loss_function(imitator_action_probabilities,expert_actions).float()
        self.loss.backward()
        self.optimizer.step()


    def get_action(self,states):
        self.total_steps += 1
        expert_action = self.get_expert_action

        if expert_action:
            return self.expert_agent.get_action(states)
        else:
            if self.configuration_parameters['Environment']['action_space_type'] == 'Discrete':
                actions= self.get_discrete_action(states)
                return actions
            else:
                return self.get_continuous_action(states)

    def get_optimal_action(self,state):
        return self.get_action(state)

    def get_discrete_action(self,states):
        action_probabilities = self.actor(states.float())
        actions = action_probabilities.argmax(dim=-1)
        actions = torch.nn.functional.one_hot(actions,self.action_space_dim)
        return actions

    def get_continuous_action(self,states):
        action_output = self.actor(states.float())
        return action_output

    def set_input_output_dim(self):
        self.actor_input_dim = self.state_space_dim
        self.actor_output_dim = self.action_space_dim
        self.configuration_parameters['Actor']['input_dim'] = self.actor_input_dim
        self.configuration_parameters['Actor']['output_dim'] = self.actor_output_dim

    def set_save_attributes(self):
        self.save_attributes = ['optimizer', 'actor', 'memory_buffer']


    def load_expert_agent(self):
        current_directory = os.getcwd()
        load_dict = torch.load(current_directory + self.expert_agent_path)
        configuration_parameters = load_dict['SavedAgent']
        expert_class = globals()[self.configuration_parameters['TrainingAgent']['expert_agent_type']]
        self.expert_agent = self.expert_agent = expert_class(configuration_parameters,self.summary_writer)
        self.expert_agent.load(load_dict = load_dict)

    def __str__(self):
        return f'<DAGGER(episodes={self.episodes_completed}, buffer_size={len(self.memory_buffer)})'