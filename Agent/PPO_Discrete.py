import os
import torch
from torch.distributions import Categorical, Normal
from Agent.PPO import PPO
from Utilities.Utils import NetworkBuilder
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer

class PPO_Discrete(PPO):
    def __init__(self, configuration_parameters,summary_writer):
        super(PPO_Discrete,self).__init__(configuration_parameters,summary_writer)

    def update(self):
        if self.episodes_completed % self.update_episodes == 0:
            batch = self.memory_buffer.get_minibatch(ppo=True)
            actions, states, rewards, next_states, end_of_episodes = self.unpack_minibatch(batch)
            batch_size = len(actions)

            actions = actions.argmax(dim=1)
            values = self.critic(states)
            next_state_values = self.critic(next_states).detach()

            advantages, returns = self.calculate_advantage_returns(values,next_state_values,rewards,end_of_episodes)
            initial_log_probabilities = self.get_logprobs(states,actions).detach()

            for _ in range(0,self.epochs):
                permutated_indices = torch.randperm(batch_size)
                for start_idx in range(0,batch_size,self.minibatch_size):
                    self.updates += 1
                    end_idx = start_idx + self.minibatch_size
                    indices = permutated_indices[start_idx:end_idx]

                    values = self.critic(states[indices])
                    log_probabilities, entropy= self.get_logprobs_entropy(states[indices],actions[indices])
                    probability_ratios = log_probabilities - initial_log_probabilities[indices]

                    surrogate_obj1 = probability_ratios*advantages[indices]
                    surrogate_obj2 = torch.clamp(probability_ratios,1-self.epsilon_clip,1+self.epsilon_clip) * advantages[indices]

                    self.optimizer.zero_grad()
                    self.actor_loss = -torch.min(surrogate_obj1,surrogate_obj2).mean()
                    self.entropy_loss = -(self.entropy_beta * entropy).mean()
                    self.critic_loss = 0.5 * self.loss_function(values, returns[indices].unsqueeze(dim=-1))
                    self.loss = self.actor_loss + self.entropy_loss + self.critic_loss
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(),self.gradient_clip)
                    self.optimizer.step()
                    if _ == 0:
                        self.summary_writer.add_scalar('Actor Loss', self.actor_loss, self.episodes_completed)
                        self.summary_writer.add_scalar('Critic Loss', self.critic_loss, self.episodes_completed)

            self.memory_buffer.reset_buffer()

    def get_action(self,state,evaluate=False):
        self.total_steps += 1
        action_probabilities = self.actor(state.float())
        #action = self.softmax(action)
        if not evaluate:
            action_index = Categorical(action_probabilities).sample()
            action = torch.nn.functional.one_hot(action_index,self.action_space_dim)
        else:
            action_index = action_probabilities.argmax(dim=-1)
            action = torch.nn.functional.one_hot(action_index,self.action_space_dim)
        return action

    def get_logprobs(self,states,actions):
        action_probabilities = self.actor(states.float())
        action_distributions = Categorical(action_probabilities)
        #actions = actions.argmax(dim=-1)`   `
        log_probabilities = action_distributions.log_prob(actions)
        return log_probabilities

    def get_logprobs_entropy(self,states,actions):
        action_probabilities = self.actor(states.float())
        action_distributions = Categorical(action_probabilities)
        #actions = actions.argmax(dim=-1)
        log_probabilities = action_distributions.log_prob(actions)
        entropy = action_distributions.entropy()
        return log_probabilities, entropy

    def __str__(self):
        return f'<PPO_Discrete(episodes={self.episodes_completed}, buffer_size={len(self.memory_buffer)})'