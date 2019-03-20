# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""

import torch
import torch.multiprocessing as mp
from env import Environment
from model import Network


class Trajectory(object):
    """class to store trajectory data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logit = []
        self.last_actions = []
        self.actor_id = None
        self.lstm_hx = None

    def append(self, state, action, reward, done, logit):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logit.append(logit)
        self.dones.append(done)

    def finish(self):
        self.states = torch.stack(self.states)
        self.rewards = torch.cat(self.rewards, 0).squeeze()
        self.actions = torch.cat(self.actions, 0).squeeze()
        self.dones = torch.cat(self.dones, 0).squeeze()
        self.logit = torch.cat(self.logit, 0)

    def cuda(self):
        self.states = self.states.cuda()
        self.actions = self.actions.cuda()
        self.dones = self.dones.cuda()
        self.lstm_hx = self.lstm_hx.cuda()
        self.rewards = self.rewards.cuda()
        self.logit = self.logit.cuda()

    @property
    def length(self):
        return len(self.rewards)

    def __repr__(self):
        return "ok"


class Actor(mp.Process):
    """Simple actor """

    def __init__(self, idx, g_net, data, env_args, action_space, length, save_path, load_path):
        super().__init__()
        self.id = idx
        self.data = data
        self.steps = 0
        self.length = length
        self.env = Environment(env_args)
        self.action_space = action_space
        self.g_net = g_net
        self.model = Network(action_space=self.action_space)
        self.hx = torch.zeros((2, 1, 256), dtype=torch.float32)
        self.rewards = 0
        self.save_path = save_path
        self.load_path = load_path

    def run(self):
        """Run the env for n steps and return a trajectory rollout."""
        done = torch.tensor(True, dtype=torch.uint8).view(1, 1)

        while True:
            print("actor {} Trajectory steps {}".format(self.id, self.steps))
            self.model.load_state_dict(self.g_net.state_dict())
            rollout = Trajectory()
            rollout.actor_id = self.id
            rollout.lstm_hx = self.hx.squeeze()
            total_reward = 0
            with torch.no_grad():
                while True:
                    if rollout.length == self.length:
                        self.rewards += total_reward
                        rollout.finish()
                        self.data.put(rollout)
                        break
                    if done:
                        self.rewards = 0
                        self.steps = 0
                        obs = self.env.reset()
                        self.hx = torch.zeros((2, 1, 256), dtype=torch.float32)
                        logits = torch.zeros((1, self.action_space), dtype=torch.float32)
                        last_action = torch.zeros((1, 1), dtype=torch.int64)
                        reward = torch.tensor(0, dtype=torch.float32).view(1, 1)
                        rollout.append(obs, last_action, reward, done, logits)
                    action, logits, hx = self.model(obs.unsqueeze(0), last_action, reward,
                                                    done, self.hx, actor=True)
                    obs, reward, done = self.env.step(action)
                    total_reward += reward
                    last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                    reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
                    done = torch.tensor(done, dtype=torch.uint8).view(1, 1)
                    rollout.append(obs, last_action, reward, done, logits.detach())
                    self.steps += 1
