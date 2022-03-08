import argparse, os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import configparser
from Trainer.Trainer import Trainer as trainer

def load_configurations(config_file_path):
    conf_parser = configparser.ConfigParser()
    conf_parser.read(config_file_path)
    conf_dict = dict()
    for section in conf_parser.sections():
        conf_dict[section] = dict()
        for key in conf_parser[section]:
            conf_dict[section][key] = eval(conf_parser[section][key])
    return conf_dict

def get_config_dict(config_file_path):
    config_dict = load_configurations(config_file_path)
    #print(config_file_path)
    current_directory = os.getcwd()
    save_folder = current_directory + config_dict['Trainer']['save_agent_folder']
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    shutil.copy(config_file_path,save_folder)
    #print(f'Making Trainer with {config_dict}')
    return config_dict

def save_config(save_folder,config_file_path):
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    shutil.copy(config_file_path,save_folder)

def plot_rewards(file_path):
    rewards_file = os.getcwd() + file_path
    rewards = np.load(rewards_file)
    print(rewards.max())
    y = np.ones((50))
    z = np.ones(len(rewards))
    x = np.convolve(rewards,y,'same') / np.convolve(z,y,'same')
    plt.plot(range(len(rewards)), x)
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Rewards')
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configuration', required=False, type=str, default = False, help='Name of .ini configuration file')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',default=False, help='Flag for evaluation session')
parser.add_argument('-f', '--folder', required = False, type=str, default = False, help='Folder containing configuration files to iterate through')
parser.add_argument('-i', '--iterate', action='store_true',help='Option for iterating through hyperparams')
parser.add_argument('-l', '--learning_rate',action='store_true', help='Flag to iterate through learning rates in a single config')
parser.add_argument('-p', '--plot_rewards', required=False, type=str, default=False, help='File path to reward vector to plot')
parser.add_argument('-t', '--target',action='store_true', help='Flag to iterate through target cloning steps')
parser.add_argument('-g', '--gamma', action='store_true', help='Flag to iterate through gamma values')
args = parser.parse_args()

assert args.configuration or args.folder or args.plot_rewards, "Missing either configuration file name, configuration folder name, or plot data file name"
if args.folder:
    config_dir = os.getcwd() + '/ParameterConfigurations/' + args.folder
    for i, filename in enumerate(os.listdir(config_dir)):
        assert filename.endswith('ini'), 'Incorrect configuration file type'
        file_path = os.path.join(config_dir, filename)
        config_dict = get_config_dict(file_path)
        trainer = trainer(config_dict)
        print('Agent {} Training Complete'.format(i))
elif args.iterate and args.configuration:
    file_path= os.getcwd() + '/ParameterConfigurations/' + args.configuration
    config_dict = get_config_dict(file_path)
    if args.learning_rate and args.target:
        assert type(config_dict['TrainingAgent']['learning_rate']) == list, 'Learning rate must be a list to iterate'
        assert type(config_dict['TrainingAgent']['target_cloning_steps']) == list, 'Target Cloning Steps must be a list to iterate'
        learning_rates = config_dict['TrainingAgent']['learning_rate']
        target_cloning_steps = config_dict['TrainingAgent']['target_cloning_steps']
        current_agent=0
        save_agent_folder = config_dict['Trainer']['save_agent_folder']
        save_agent_file_name = config_dict['Trainer']['save_agent_file_name']
        for learning_rate in learning_rates:
            config_dict['TrainingAgent']['learning_rate'] = learning_rate
            for target_steps in target_cloning_steps:
                config_dict['TrainingAgent']['target_cloning_steps'] = target_steps
                config_dict['Trainer']['save_agent_folder'] = save_agent_folder + str(current_agent)
                config_dict['Trainer']['save_agent_file_name'] = save_agent_file_name + str(current_agent)
                training_run = trainer(config_dict)
                print('Agent {} has completed Trainig'.format(current_agent))
                current_agent += 1
    elif args.gamma:
        assert type(config_dict['TrainingAgent']['gamma']) == list, 'Gamma must be a list to iterate'
        gammas = config_dict['TrainingAgent']['gamma']
        current_agent=0
        save_agent_folder = config_dict['Trainer']['save_agent_folder']
        save_agent_file_name = config_dict['Trainer']['save_agent_file_name']
        if type(config_dict['Trainer']['seed']) == list:
            seeds = config_dict['Trainer']['seed']
            for seed in seeds:
                config_dict['Trainer']['seed'] = seed
                for gamma in gammas:
                    config_dict['TrainingAgent']['gamma'] = gamma
                    config_dict['Trainer']['save_agent_folder'] = save_agent_folder + str(current_agent)
                    config_dict['Trainer']['save_agent_file_name'] = save_agent_file_name + str(current_agent)
                    training_run = trainer(config_dict)
                    print('Agent {} has completed Trainig'.format(current_agent))
                    current_agent += 1

elif args.plot_rewards:
    plot_rewards(args.plot_rewards)
else:
    file_path= os.getcwd() + '/ParameterConfigurations/' + args.configuration
    #print('File path: {}'.format(file_path))
    config_dict = get_config_dict(file_path)
    trainer = trainer(config_dict)

print('Training Session is complete')