#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


# class PPORecurrentAgent(BaseAgent):
#     def __init__(self, config):
#         BaseAgent.__init__(self, config)
#         self.config = config
#         self.task = config.task_fn()
#         if config.network:
#             self.network = config.network
#         else:
#             self.network = config.network_fn()
#         self.network.to(torch.device('cuda'))
#         self.opt = config.optimizer_fn(self.network.parameters())
#         self.total_steps = 0
#         self.recurrent_states = None
#         self.states = self.task.reset()
#         self.done = True

#     def step(self):
#         config = self.config
#         storage = Storage(config.rollout_length)
#         states = self.states
#         for _ in range(config.rollout_length):

#             with torch.no_grad():
#                 if self.done:
#                     prediction, self.recurrent_states = self.network(states)
#                 else:
#                     prediction, self.recurrent_states = self.network(states, self.recurrent_states)

#             self.done = False
#             next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
#             self.record_online_return(info)
#             rewards = config.reward_normalizer(rewards)
#             storage.add(prediction)
#             storage.add({'r': tensor(rewards).unsqueeze(-1),
#                          'm': tensor(1 - terminals).unsqueeze(-1),
#                          's': tensor(states),
#                          'rs': self.recurrent_states})
#             states = next_states
#             self.total_steps += config.num_workers

#         self.states = states
#         with torch.no_grad():
#             prediction, self.recurrent_states = self.network(states, self.recurrent_states)
#         storage.add(prediction)
#         storage.placeholder()

#         advantages = tensor(np.zeros((config.num_workers, 1)))
#         returns = prediction['v'].detach()
#         for i in reversed(range(config.rollout_length)):
#             returns = storage.r[i] + config.discount * storage.m[i] * returns
#             if not config.use_gae:
#                 advantages = returns - storage.v[i].detach()
#             else:
#                 td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
#                 advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
#             storage.adv[i] = advantages.detach()
#             storage.ret[i] = returns.detach()

#         states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
#         actions = actions.detach()
#         log_probs_old = log_probs_old.detach()
#         advantages = (advantages - advantages.mean()) / advantages.std()

#         for _ in range(config.optimization_epochs):
#             sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
#             for batch_indices in sampler:
#                 batch_indices = tensor(batch_indices).long()
#                 sampled_states = states[batch_indices]
#                 sampled_actions = actions[batch_indices]
#                 sampled_log_probs_old = log_probs_old[batch_indices]
#                 sampled_returns = returns[batch_indices]
#                 sampled_advantages = advantages[batch_indices]

#                 prediction = self.network(sampled_states, sampled_actions)
#                 ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
#                 obj = ratio * sampled_advantages
#                 obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
#                                           1.0 + self.config.ppo_ratio_clip) * sampled_advantages
#                 policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

#                 value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

#                 self.opt.zero_grad()
#                 (policy_loss + value_loss).backward()
#                 nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
#                 self.opt.step()


#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# TODO:
# - plot average rewards in matplotlib
# - look at when entropy loss is recorded


from ..network import *
from ..component import *
from .BaseAgent import *

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance

import numpy
import numpy.random

import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPORecurrentAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config #config file, contains hyperparameters and other info
        self.task = config.task_fn() #gym environment

        if config.network: #nnet used
            self.network = config.network
        else:
            self.network = config.network_fn()
        self.network.to(device)

        self.optimizer = config.optimizer_fn(self.network.parameters()) #optimization function
        self.total_steps = 0
        self.recurrent_states = None
        self.first_recurrent_states = None
        self.states = self.task.reset()
        self.recurrence = config.recurrence
        self.done = True

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)

        states = self.states
        self.first_recurrent_states = self.recurrent_states
        for _ in range(config.rollout_length):
            #put states and recurrent states into storage
            if self.done:
                cleared = [None for i in range(self.config.num_workers)]
                self.recurrent_states = [cleared, cleared]
                storage.add({'rs': self.recurrent_states})
            else:
                rstates = list(self.recurrent_states)
                for i, state in enumerate(rstates):
                    rstates[i] = state.detach()
                storage.add({'rs': rstates})

            #run the neural net once to get prediction
            start = time.time()
            if self.done:
                prediction, self.recurrent_states = self.network(states)
            else:
                prediction, self.recurrent_states = self.network(states, self.recurrent_states)
            end = time.time()
            self.logger.add_scalar('forward_pass_time', end-start, self.total_steps)

            self.done = False

            #step the environment with the action determined by the prediction
            start = time.time()
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            end = time.time()
            self.logger.add_scalar('env_step_time', end-start, self.total_steps)

            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)

            #add everything to storage
            storage.add(prediction)
            storage.add({'s': states})
            storage.add({'r': tensor(rewards).unsqueeze(-1).to(device),
                         'm': tensor(1 - terminals).unsqueeze(-1).to(device)})
            states = next_states

            self.total_steps += config.num_workers

        self.states = states

        prediction, self.recurrent_states = self.network(states)

        #TODO:This could possibly be an issue, would this prediction ever be used?
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1))).to(device)
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        log_probs_old, values, returns, advantages= storage.cat(['log_pi_a', 'v', 'ret', 'adv'])
        log_probs_old = log_probs_old.detach()
        values = values.detach()
        states = storage.s
        rc_states = storage.rs

        advantages = (advantages - advantages.mean()) / advantages.std()

        self.logger.add_scalar('advantages', advantages.mean(), self.total_steps)


        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(len(states)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()

                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                sampled_rc_states = [rc_states[i] for i in batch_indices]
                sampled_states = []
                for i in batch_indices:
                    sampled_states = sampled_states + list(states[i])

                prediction, _ = self.network(sampled_states, sampled_rc_states)

                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

                self.logger.add_scalar('entropy_loss', prediction['ent'].mean(), self.total_steps)
                self.logger.add_scalar('policy_loss', policy_loss, self.total_steps)
                self.logger.add_scalar('value_loss', value_loss, self.total_steps)
