import os
import torch
import torch.nn as nn
import torch.optim as optim
from Utilities.Utils import NetworkBuilder
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer




class Agent(nn.Module):
    def __init__(self, configuration_parameters, summary_writer):
        super(Agent,self).__init__()
        self.configuration_parameters = configuration_parameters
        self.set_object_parameters()
        torch.manual_seed(self.seed)
        self.total_steps = 0
        self.episodes_completed = 0
        self.loss_function = getattr(nn,self.loss_function,None)()
        self.network = globals()['NetworkBuilder']
        self.summary_writer = summary_writer

    def adjust_learning_rate(self,learning_rate_adjustment):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate_adjustment
        print('Learning Rate has been adjusted')

    def get_action(self,state):
        pass

    def get_optimal_action(self,state):
        pass

    def update(self, buffer_sample):
        pass

    def set_input_outpute_dim(self):
        pass

    def make_buffer(self):
        buffer_class = globals()[self.configuration_parameters['MemoryBuffer']['type']]
        self.memory_buffer = buffer_class(self.configuration_parameters)

    def unpack_minibatch(self,minibatch):
        return minibatch[0], minibatch[1], minibatch[2], minibatch[3], minibatch[4]

    def set_object_parameters(self):
        if 'SavedAgent' in self.configuration_parameters.keys():
            for key, value in self.configuration_parameters['SavedAgent'].items():
                setattr(self,key,value)
        else:
            for key, value in self.configuration_parameters['TrainingAgent'].items():
                setattr(self,key,value)

        self.action_space_dim = self.configuration_parameters['Environment']['action_space_dim']
        self.state_space_dim = self.configuration_parameters['Environment']['state_space_dim']

    def save(self,file_name):
        save_dict = dict()
        current_directory = os.getcwd()

        for attribute in self.save_attributes:
            attribute_object = getattr(self,attribute)
            if hasattr(attribute_object,'state_dict'):
                save_dict[attribute] = attribute_object.state_dict()
            else:
                save_dict[attribute] = attribute_object

        save_dict['SavedAgent'] = self.configuration_parameters
        #torch.save(save_dict,current_directory + f'/Saved_Agents/{file_name}.pth')
        torch.save(save_dict,current_directory + f'{file_name}.pth')

    def load(self,file_name=None, load_dict=None):
        assert file_name or load_dict, "Need to specify state dictionary or file name for state dictionary to load from"

        if file_name != None:
            current_directory = os.getcwd()
            load_dict = torch.load(current_directory + f'/Saved_Agents/{file_name}.pth')

        for attribute in load_dict.keys():
            if hasattr(self,attribute):
                attribute_object = getattr(self,attribute)
                if hasattr(attribute_object,'state_dict'):
                    attribute_object.load_state_dict(load_dict[attribute])
                else:
                    setattr(self,attribute,load_dict[attribute])
