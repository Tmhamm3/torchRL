import torch
import torch.nn as nn
from torch.nn import Linear, Sequential

def NetworkBuilder(configuration_parameters : dict):
    '''
        Current: Configured to handle Linear Networks
        Future: Network Builder to create networks with different combinations of Network types, i.e... ANN, RNN, CNN
    '''
    torch.manual_seed(configuration_parameters['seed'])
    layers = list()
    layer_type = globals()[configuration_parameters['layer_type']]
    network_type = globals()[configuration_parameters['network_type']]
    input_dim = configuration_parameters['input_dim']
    output_dim = configuration_parameters['output_dim']
    layer_sizes = configuration_parameters['layer_sizes']
    activation_functions = configuration_parameters['activation_functions']

    initial_layer = layer_type(input_dim,layer_sizes[0])
    initial_layer.weight.data.normal_(0,0.01)
    layers.append(layer_type(input_dim,layer_sizes[0]))
    layers.append(getattr(torch.nn,activation_functions[0], None)())
    input_dim = layer_sizes[0]

    # Function to handle middle layers of the Network
    for layer_size, activation_function in tuple(zip(layer_sizes[1:], activation_functions[1:])):
        temp_layer = layer_type(input_dim,layer_size)
        temp_layer.weight.data.normal_(0,0.01)
        layers.append(temp_layer)
        temp_activation_type= getattr(nn,activation_function,None)
        temp_activation_function = temp_activation_type()
        layers.append(temp_activation_function)
        input_dim = layer_size

    temp_layer = layer_type(input_dim,output_dim)
    temp_layer.weight.data.normal_(0,0.01)
    layers.append(temp_layer)

    if configuration_parameters['output_function'] is not None:
        if configuration_parameters['output_function'] == 'Softmax':
            layers.append(getattr(torch.nn,configuration_parameters['output_function'],None)(dim=-1))
        else:
            layers.append(getattr(torch.nn,configuration_parameters['output_function'],None)())
            
    return Sequential(*layers)