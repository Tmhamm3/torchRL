import os
import torch
from torch.distributions import Categorical, Normal
from Agent.PPO import PPO
from Utilities.Utils import NetworkBuilder
from MemoryBuffer.SAMemoryBuffer import SAMemoryBuffer

class PPO_Continuous(PPO):
    def __init__(self, configuration_parameters,summary_writer):
        super(PPO_Continuous,self).__init__(configuration_parameters,summary_writer)

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
            #permutated_indices = torch.randperm(batch_size)

            for _ in range(0,self.epochs):
                permutated_indices = torch.randperm(batch_size)
                for start_idx in range(0,batch_size,self.minibatch_size):
                    self.updates += 1
                    end_idx = start_idx + self.minibatch_size
                    indices = permutated_indices[start_idx:end_idx]

                    values = self.critic(states[indices])
                    if self.cdf_penalty:
                        log_probabilities, entropy, cdf_loss= self.get_logprobs_entropy(states[indices],actions[indices],cdf=True)
                    else:
                        log_probabilities, entropy= self.get_logprobs_entropy(states[indices],actions[indices])
                    probability_ratios = torch.exp(log_probabilities - initial_log_probabilities[indices])

                    surrogate_obj1 = probability_ratios*advantages[indices]
                    surrogate_obj2 = torch.clamp(probability_ratios,1-self.epsilon_clip,1+self.epsilon_clip) * advantages[indices]

                    self.optimizer.zero_grad()
                    self.actor_loss = -torch.min(surrogate_obj1,surrogate_obj2).mean()
                    self.entropy_loss = -(self.entropy_beta * entropy).mean()
                    self.critic_loss = 0.5 * self.loss_function(values, returns[indices].unsqueeze(dim=-1))
                    self.loss = self.actor_loss + self.entropy_loss + self.critic_loss
                    if self.cdf_penalty:
                        self.loss += cdf_loss.mean()
                    self.loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(),self.gradient_clip)
                    self.optimizer.step()
                    if _ == 0:
                        self.summary_writer.add_scalar('Actor Loss', self.actor_loss, self.episodes_completed)
                        self.summary_writer.add_scalar('Critic Loss', self.critic_loss, self.episodes_completed)
                        self.summary_writer.add_scalar('Mean', self.mean, self.episodes_completed)
                        self.summary_writer.add_scalar('Std', self.std, self.episodes_completed)
                        if self.cdf_penalty:
                            self.summary_writer.add_scalar('Cdf Penalty', cdf_loss.mean(), self.episodes_completed)

            self.memory_buffer.reset_buffer()

    def get_action(self,state,evaluate=False,squeeze=False):
        self.total_steps += 1
        distribution = self.get_distribution(state,squeeze)
        #mean = self.mu(state.float())
        #log_std = self.log_std.expand_as(mean)
        #action = Normal(mean,log_std.exp()).sample()
        if not evaluate:
            action = distribution.sample()
            self.summary_writer.add_scalar('Action Value',action,self.total_steps)
        else:
            action = distribution.mean
        action = torch.clamp(action,-1,1)
        return action

    def get_distribution(self,states,squeeze=True,cdf=False):
        if squeeze:
            mean = self.mu(states.float()).squeeze(dim=-1)
        else:
            mean = self.mu(states.float())
        std = self.log_std.expand_as(mean).exp()
        #std = mean[:,self.action_space_dim:].squeeze(dim=-1)
        #mean = mean[:,:self.action_space_dim].squeeze(dim=-1)
        self.mean = mean.mean()
        self.std = std.mean()
        distribution = Normal(mean,std)
        if cdf:
            if self.mean >=0:
                cdf_loss = 1 - distribution.cdf(self.mean)
            else:
                cdf_loss = distribution.cdf(self.mean)
            return distribution, cdf_loss
        else:
            return distribution

    def get_logprobs(self,states, actions):
        #mean = self.mu(states.float()).squeeze(dim=-1)
        #std = self.log_std.expand_as(mean).exp()
        #log_probs = Normal(mean,std).log_prob(actions)
        distribution = self.get_distribution(states)
        log_probs = distribution.log_prob(actions)
        return log_probs


    def get_logprobs_entropy(self,states,actions,cdf=False):
        #mean = self.mu(states.float()).squeeze(dim=-1)
        #std = self.log_std.expand_as(mean).exp()
        if cdf:
            distribution, cdf_loss = self.get_distribution(states,cdf=cdf)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()

            return log_probs, entropy, cdf_loss
        else:
            distribution = self.get_distribution(states,cdf)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return log_probs, entropy

    def __str__(self):
        return f'<PPO_Continuous(episodes={self.episodes_completed}, buffer_size={len(self.memory_buffer)})'