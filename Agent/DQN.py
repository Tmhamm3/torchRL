import torch
import copy
from Agent.Agent import Agent
from Utilities.Utils import NetworkBuilder
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer

class DQN(Agent):
    def __init__(self, configuration_parameters,summary_writer):
        super(DQN,self).__init__(configuration_parameters,summary_writer)
        self.set_input_output_dim()
        self.set_save_attributes()
        self.make_buffer()
        self.QNet= self.network(self.configuration_parameters['NeuralNetworks'])
        self.targetQNet = copy.deepcopy(self.QNet)
        self.optimizer = getattr(torch.optim,self.optimizer,None)(params=self.parameters(),lr=self.learning_rate)

    def get_action(self,state:torch.tensor,evaluate=False) -> torch.tensor:
        self.total_steps+=1
        epsilon_value = torch.rand(1).item()
        if (not evaluate) and (self.total_steps <= self.initial_exploration_steps or epsilon_value <= self.epsilon):
            return self.random_action()
        else:
            return self.q_action(state)

    def get_optimal_action(self,state):
        return self.q_action(state)

    def q_action(self,state:torch.tensor, target=False) -> torch.tensor:
        q_values = self.QNet(state.float())
        action_index = q_values.argmax(dim=-1)
        action = torch.nn.functional.one_hot(action_index,self.action_space_dim)
        return action

    def random_action(self) -> torch.tensor:
        action = torch.zeros(self.action_space_dim).float()
        action_index = torch.randint(0,self.action_space_dim,(1,)).item()
        action[action_index] = 1
        return action

    def set_input_output_dim(self):
        self.input_dim = self.state_space_dim
        self.output_dim = self.action_space_dim
        self.configuration_parameters['NeuralNetworks']['input_dim'] = self.input_dim
        self.configuration_parameters['NeuralNetworks']['output_dim'] = self.output_dim

    def update(self):
        minibatch = self.memory_buffer.get_minibatch()
        actions, states, rewards, next_states, end_of_episodes = self.unpack_minibatch(minibatch)
        actions = actions.argmax(dim=-1)

        q_values = self.QNet(states).gather(1,actions.unsqueeze(dim=-1)).squeeze(-1)
        target_q_values = self.targetQNet(next_states).max(dim=1)[0].detach()
        target_q_values[end_of_episodes.bool()] = 0
        expected_values = rewards.squeeze(dim=-1) + (self.gamma * target_q_values)

        self.optimizer.zero_grad()
        self.loss = self.loss_function(q_values,expected_values)
        self.loss.backward()
        self.optimizer.step()
        self.summary_writer.add_scalar('Loss', self.loss, self.episodes_completed)
        self.check_target_copy()

    def check_target_copy(self):
        if self.total_steps % self.target_cloning_steps == 0:
            self.targetQNet = copy.deepcopy(self.QNet)

    def set_save_attributes(self):
        self.save_attributes = ['optimizer', 'QNet', 'targetQNet','memory_buffer']

    def __str__(self):
        return f'<DQN(episodes={self.episodes_completed}, buffer_size={len(self.memory_buffer)})'