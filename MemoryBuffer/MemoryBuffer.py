from abc import ABC
import torch

class MemoryBuffer(ABC):
    def __init__(self, configuration_parameters:dict):
        """
            Expected Parameters:
                buffer_size: Max number of steps in buffer
                action_space_dim: Dimensionality of the environment action space
                state_space_dim: Dimensionality of the environment observation space
                reward_space_dim: Dimensionality of the emvironment reward space
        """
        self.configuration_parameters = configuration_parameters
        self.set_object_parameters()
        self.current_index = 0
        self.buffer_fill = 0
        self.__initialize_buffer

    def __initialize_buffer(self):
        pass

    def push(self):
        pass

    def get_batch(self):
        pass

    def set_object_parameters(self):
        for key, value in self.configuration_parameters['MemoryBuffer'].items():
            setattr(self,key,value)
        self.action_space_dim = self.configuration_parameters['Environment']['action_space_dim']
        self.state_space_dim = self.configuration_parameters['Environment']['state_space_dim']
        self.reward_space_dim = self.configuration_parameters['Environment']['reward_space_dim']