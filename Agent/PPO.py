import os
import torch
from torch.distributions import Categorical, Normal
from Agent.Agent import Agent
from Utilities.Utils import NetworkBuilder
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer

class PPO(Agent):
    def __init__(self, configuration_parameters,summary_writer):
        super(PPO,self).__init__(configuration_parameters,summary_writer)
        self.updates = 0
        self.set_input_output_dim()
        self.set_save_attributes()
        self.make_buffer()
        self.set_networks_update()
        self.actor_loss = 0
        self.critic_loss = 0

    def update(self):
        pass 

    def get_optimal_action(self,state):
        return self.get_action(state,evaluate=True)

    def get_action(self,state,evaluate=False,squeeze=False):
        pass


    def get_distribution(self,states,squeeze=True,cdf=False):
        pass

    def get_logprobs(self,states, actions):
        pass

    def get_logprobs_entropy(self,states,actions,cdf=False):
        pass

    def calculate_advantage_returns(self,values, next_state_values,rewards,end_of_episodes):
        returns = []
        advantages = []
        delta= 0
        gae = 0
        for i in reversed(range(len(rewards))):
            if end_of_episodes[i]:
                delta = rewards[i]-values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_state_values[i]  - values[i]
                gae = delta + (self.gamma * self.advantage_lambda * gae)
            returns.insert(0,gae+values[i])
            advantages.insert(0,gae)
        #Format discounted rewards and advantages for torch use
        returns = torch.tensor(returns).float()
        advantages = torch.tensor(advantages).float()
        #Normalize the advantages
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages))
        #if self.normalized_reward:
            #returns = (returns -torch.mean(returns)) / (torch.std(returns) + 1e-5)
        return advantages, returns

    def set_networks_update(self):
        assert self.action_space_type == 'Discrete' or self.action_space_type == 'Continuous', 'Incompatible Action Space Type'
        if self.action_space_type == 'Discrete':
            self.actor = self.network(self.configuration_parameters['Actor'])
            self.critic = self.network(self.configuration_parameters['Critic'])
            self.optimizer = getattr(torch.optim, self.optimizer,None)(params=self.parameters(),lr=self.learning_rate)
        elif self.action_space_type == 'Continuous':
            self.mu = self.network(self.configuration_parameters['Mu'])
            self.critic = self.network(self.configuration_parameters['Critic'])
            self.log_std = torch.nn.Parameter(torch.zeros(self.action_space_dim))
            self.optimizer = getattr(torch.optim,self.optimizer,None)(params=self.parameters(),lr=self.learning_rate)

    def set_input_output_dim(self):
        self.actor_input_dim = self.state_space_dim
        self.actor_output_dim = self.action_space_dim
        self.critic_input_dim = self.state_space_dim
        self.critic_output_dim = 1
        self.action_space_type = self.configuration_parameters['Environment']['action_space_type']
        if self.action_space_type == 'Discrete':
            self.configuration_parameters['Actor']['input_dim'] = self.actor_input_dim
            self.configuration_parameters['Actor']['output_dim'] = self.actor_output_dim
        else:
            self.configuration_parameters['Mu']['input_dim'] = self.actor_input_dim
            self.configuration_parameters['Mu']['output_dim'] = self.actor_output_dim
        self.configuration_parameters['Critic']['input_dim'] = self.critic_input_dim
        self.configuration_parameters['Critic']['output_dim'] = self.critic_output_dim

    def set_save_attributes(self):
        if self.action_space_type =='Discrete':
            self.save_attributes = ['optimizer', 'critic', 'actor', 'memory_buffer']
        else:
            self.save_attributes = ['optimizer', 'critic', 'mu', 'log_std', 'memory_buffer']

    def __str__(self):
        return f'<PPO(episodes={self.episodes_completed}, buffer_size={len(self.memory_buffer)})'