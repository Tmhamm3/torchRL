import torch
import torch.nn as nn

class SequentialArtificialNeuralNetwork(nn.Module):
    def __init__(self,configuration_parameters):
        super(SequentialArtificialNeuralNetwork, self).__init__()
        self.configuration_parameters = configuration_parameters
        self.set_object_parameters()
        assert len(self.layer_sizes) == len(self.activation_functions), 'Number of layers must match number of activation functions'

    def set_object_parameters(self):
        for key, value in self.configuration_parameters['NeuralNetworks'].items():
            print(f'Key: {key}')
            setattr(self,key,value)