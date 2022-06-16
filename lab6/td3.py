'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))

class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        ## TODO ##
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.tanh(self.fc3(x))
        return out


# class CriticNet(nn.Module):
#     def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
#         super().__init__()
#         h1, h2 = hidden_dim
#         self.critic_head1 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, h1),
#             nn.ReLU(),
#         )
#         self.critic1 = nn.Sequential(
#             nn.Linear(h1, h2),
#             nn.ReLU(),
#             nn.Linear(h2, 1),
#         )

#         h3, h4 = hidden_dim
#         self.critic_head2 = nn.Sequential(
#             nn.Linear(state_dim + action_dim, h3),
#             nn.ReLU(),
#         )
#         self.critic2 = nn.Sequential(
#             nn.Linear(h3, h4),
#             nn.ReLU(),
#             nn.Linear(h4, 1),
#         )

#     def forward(self, x, action):
#         x1 = self.critic_head1(torch.cat([x, action], dim=1))
#         x2 = self.critic_head2(torch.cat([x, action], dim=1))
#         return self.critic1(x1) , self.critic2(x2)

#     def get_Q(self, x, action):
#         x = self.critic_head1(torch.cat([x, action], dim=1))
#         return self.critic1(x)

class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)




class TD3:
    def __init__(self, args, max_action):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net1 = CriticNet().to(args.device)
        self._critic_net2 = CriticNet().to(args.device)


        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net1 = CriticNet().to(args.device)
        self._target_critic_net2 = CriticNet().to(args.device)

        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net1.load_state_dict(self._critic_net1.state_dict())
        self._target_critic_net2.load_state_dict(self._critic_net2.state_dict())
        ## TODO ##
        self._actor_opt = optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt1 = optim.Adam(self._critic_net1.parameters(), lr=args.lrc)
        self._critic_opt2 = optim.Adam(self._critic_net2.parameters(), lr=args.lrc)
        
        # action noise
        self._action_noise = GaussianNoise(dim=2)
        
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

        self.max_action = max_action

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        with torch.no_grad():
            action = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device)).cpu().numpy().squeeze()
            if(noise):
                action += self._action_noise.sample()       
            return action


    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self, args, episode):
        # update the behavior networks
        self._update_behavior_network(self.gamma, args, episode)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,self.tau)
        self._update_target_network(self._target_critic_net1, self._critic_net1,self.tau)
        self._update_target_network(self._target_critic_net2, self._critic_net2,self.tau)

    def _update_behavior_network(self, gamma, args, epoch):
        # actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, 
                                                        # self._target_actor_net, self._target_critic_net
        # actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        q1_value = self._critic_net1(state, action)
        q2_value = self._critic_net2(state, action)
        with torch.no_grad():
        ### Target Policy Smoothing ###
           noise = torch.ones_like(action).data.normal_(0, args.policy_noise).to(self.device).clamp(-args.noise_clip, args.noise_clip)
           a_next = (self._target_actor_net(next_state) + noise).clamp(-self.max_action, self.max_action)
           ### Clipped Double-Q Learning ###
           q1_next = self._target_critic_net1(next_state, a_next)
           q2_next = self._target_critic_net2(next_state, a_next)
           q_next = torch.min(q1_next, q2_next)
           q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q1_value, q_target)
        critic_loss2 = criterion(q2_value, q_target) 

        # optimize critic
        self._actor_net.zero_grad()
        self._critic_net1.zero_grad()
        self._critic_net2.zero_grad()
        critic_loss1.backward()
        critic_loss2.backward()
        self._critic_opt1.step()
        self._critic_opt2.step()

        ### “Delayed” Policy Updates ###
        if epoch % args.policy_delay == 0:
            ## update actor ##
            # actor loss
            ## TODO ##
            action = self._actor_net(state)
            actor_loss = -self._critic_net1(state, action).mean()

            # optimize actor
            self._actor_net.zero_grad()
            self._critic_net1.zero_grad()
            self._critic_net2.zero_grad()
            actor_loss.backward()
            self._actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            target.data.copy_(target.data * (1 - tau) + behavior.data * tau)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic1': self._target_critic_net1.state_dict(),
                    'target_critic2': self._target_critic_net2.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt1': self._critic_opt1.state_dict(),
                    'critic_opt2': self._critic_opt2.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net1.load_state_dict(model['critic1'])
        self._critic_net2.load_state_dict(model['critic2'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net1.load_state_dict(model['target_critic1'])
            self._target_critic_net2.load_state_dict(model['target_critic2'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt1.load_state_dict(model['critic_opt1'])
            self._critic_opt2.load_state_dict(model['critic_opt2'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(args, episode)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()

        for e in itertools.count(start=1):

            env.render()
            # select action
            action = agent.select_action(state, noise=False)
            # execute action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            if(done):
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print("total reward : {0:.2f}".format(total_reward))
                rewards.append(total_reward)
                break

    print('Average Reward', np.mean(rewards))
    env.close()


def main():

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='td3.pth')
    parser.add_argument('--logdir', default='log/td3_'+ TIMESTAMP)
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=50000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)

    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    max_action = float(env.action_space.high[0])
    agent = TD3(args, max_action)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
