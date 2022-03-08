import torch
from MemoryBuffer.MemoryBuffer import MemoryBuffer

class SAMemoryBuffer(MemoryBuffer):
    def __init__(self, configuration_parameters:dict):
        super(SAMemoryBuffer, self).__init__(configuration_parameters)
        self.__initialize_buffer()
        torch.manual_seed(self.seed)


    def __initialize_buffer(self):
        self.actions = torch.zeros(self.max_size, self.action_space_dim).float()
        self.states = torch.zeros(self.max_size, self.state_space_dim).float()
        self.rewards = torch.zeros(self.max_size, self.reward_space_dim).float()
        self.next_states = torch.zeros(self.max_size, self.state_space_dim).float()
        self.end_of_episodes = torch.ByteTensor(self.max_size)

    def store(self, actions, states, rewards, next_states, end_of_episodes, steps=1):
        if self.current_index + steps >= self.max_size:
            tail_size = self.max_size - self.current_index
            head_size = steps - tail_size
            self.actions[self.current_index:,:] = actions[:tail_size]
            self.states[self.current_index:,:] = states[:tail_size]
            self.rewards[self.current_index:,:] = rewards [:tail_size]
            self.next_states[self.current_index:,:] = next_states[:tail_size]
            self.end_of_episodes[self.current_index:] = end_of_episodes[:tail_size]

            self.actions[:head_size,:] = actions[tail_size-1:]
            self.states[:head_size,:] = states[tail_size-1:]
            self.rewards[:head_size,:] = rewards[tail_size-1:]
            self.next_states[:head_size,:] = next_states[tail_size-1:]
            self.end_of_episodes[head_size] = end_of_episodes[tail_size-1:]
            self.current_index = head_size
        else:
            self.actions[self.current_index:self.current_index+steps,:] = actions
            self.states[self.current_index:self.current_index+steps,:] = states
            self.rewards[self.current_index:self.current_index+steps,:] = rewards
            self.next_states[self.current_index:self.current_index+steps,:] = next_states
            self.end_of_episodes[self.current_index:self.current_index+steps] = end_of_episodes
            self.current_index += steps

        if self.buffer_fill < self.max_size:
            self.buffer_fill += steps

    def get_minibatch(self, ppo=False):
        if not ppo:
            #indices = torch.randint(0,min(self.buffer_fill,self.max_size),(self.minibatch_size,))
            indices = torch.randperm(self.buffer_fill)[:self.minibatch_size]
            actions = self.actions[indices,:]
            states = self.states[indices,:]
            rewards = self.rewards[indices,:]
            next_states = self.next_states[indices,:]
            end_of_episodes = self.end_of_episodes[indices]
            return [actions,states,rewards,next_states,end_of_episodes]
        else:
            return [self.actions[:self.buffer_fill], self.states[:self.buffer_fill], self.rewards[:self.buffer_fill], self.next_states[:self.buffer_fill],self.end_of_episodes[:self.buffer_fill]]

    def reset_buffer(self):
        self.current_index = 0
        self.buffer_fill = 0
        self.actions = torch.zeros(self.max_size, self.action_space_dim).float()
        self.states = torch.zeros(self.max_size, self.state_space_dim).float()
        self.rewards = torch.zeros(self.max_size, self.reward_space_dim).float()
        self.next_states = torch.zeros(self.max_size, self.state_space_dim).float()
        self.end_of_episodes = torch.ByteTensor(self.max_size)


    def __len__(self):
        '''Length of sample in the buffer'''
        return self.buffer_fill