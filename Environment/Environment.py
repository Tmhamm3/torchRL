from abc import ABC
import gym, torch
from tensorboardX import SummaryWriter
import os


class Environment(ABC):
    def __init__(self, configuration_parameters):
        self.configuration_parameters = configuration_parameters
        self.set_object_parameters()
        torch.manual_seed(self.seed)
        self.episode_length = 0
        self.episodes_completed = 0
        self.episode_reward = 0
        self.steps_total = 0
        self.set_summary_writer()


    def step(self):
        pass

    def set_acs(self):
        pass

    def set_obs(self):
        pass

    def render(self,render):
        pass

    def set_object_parameters(self):
        for key, value in self.configuration_parameters['Environment'].items():
            setattr(self,key,value)
        if hasattr(self,'lr_thresholds') or hasattr(self,'evaluation_threshold'):
            self.threshold = True
            assert hasattr(self,'eval_threshold_episodes'), 'Missing threshold episodes attribute'
            self.threshold_tracker = torch.zeros(self.eval_threshold_episodes)
            self.threshold_index = 0
        if hasattr(self,'lr_thresholds'):
            self.lr_idx = 0
            self.lr_size = len(self.lr_thresholds)

    def check_episode_threshold(self):
        self.threshold_tracker[self.threshold_index] = self.episode_reward
        self.threshold_index = (self.threshold_index +1) % self.threshold_episodes
        print('Threshold Mean: {}'.format(self.threshold_tracker.mean()))
        if hasattr(self,'eval_threshold_episodes') and self.threshold_tracker.mean() >= self.evaluation_threshold:
            return 'Threshold met'
        elif hasattr(self,'lr_thresholds'):
            if self.lr_idx < self.lr_size and self.threshold_tracker.mean() >= self.lr_thresholds[self.lr_idx]:
                self.lr_idx += 1
                return 'Adjust Learning Rate'
        else:
            return 'Keep Playing'

    def episode_over(self):
        if self.end_of_episode:
            print(f'Episode {self.episodes_completed} Reward: {self.episode_reward}')
            #if self.threshold:
                #self.world_info = self.check_episode_threshold(self.episode_reward)
            self.check_thresholds()
            self.episodes_completed+=1
            self.summary_writer.add_scalar('episode_reward', self.episode_reward, self.episodes_completed)
            self.summary_writer.add_scalar('episode_length', self.episode_length, self.episodes_completed)
            self.episode_reward = 0
            self.episode_length = 0
            self.state = self.environment.reset()

    def check_thresholds(self):
        if hasattr(self,'eval_threshold_episodes'):
            self.world_info = self.check_episode_threshold(self.episode_reward)
            if self.world_info =='Threshold met':
                return self.world_info
        if hasattr(self,'learning_rate_threshold'):
            self.world_info = self.check_episode_threshold(self.episode_reward,lr=True)


    def set_summary_writer(self):
        folder = os.getcwd() + self.configuration_parameters['Trainer']['save_agent_folder']
        if not os.path.isdir(folder):
            os.mkdir(folder)
        summary_writer_directory = folder + '/tensorboard'
        #summary_writer_directory = os.getcwd() + folder + '/tensorboard/'
        if not os.path.isdir(summary_writer_directory):
            os.mkdir(summary_writer_directory)
        self.summary_writer = SummaryWriter(logdir=summary_writer_directory)