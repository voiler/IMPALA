# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""

import torch
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


def reset(conn):
    conn.send([0, 0])
    return conn.recv()


def step(conn, actions):
    conn.send([1, actions])
    return conn.recv()


def close(conn):
    conn.send([2, None])
    conn.close()
    print("Environment closed")


def actor(idx, ps, data, conn, args):
    """Simple actor """
    steps = 0
    length = args.length
    action_space = args.action_space
    model = Network(action_space=action_space)
    hx = torch.zeros((2, 1, 256), dtype=torch.float32)
    # save_path = args.save_path
    # load_path = args.load_path

    """Run the env for n steps and return a trajectory rollout."""
    done = torch.tensor(True, dtype=torch.bool).view(1, 1)
    rewards = 0
    while True:
        print("Actor: {} Steps: {} Reward: {}".format(idx, steps, rewards))
        model.load_state_dict(ps.pull())
        rollout = Trajectory()
        rollout.actor_id = idx
        rollout.lstm_hx = hx.squeeze()
        total_reward = 0
        with torch.no_grad():
            while True:
                if rollout.length == length:
                    rewards += total_reward
                    rollout.finish()
                    data.put(rollout)
                    break
                if done:
                    rewards = 0.
                    steps = 0
                    obs = reset(conn)
                    hx = torch.zeros((2, 1, 256), dtype=torch.float32)
                    logits = torch.zeros((1, action_space), dtype=torch.float32)
                    last_action = torch.zeros((1, 1), dtype=torch.int64)
                    reward = torch.tensor(0, dtype=torch.float32).view(1, 1)
                    rollout.append(obs, last_action, reward, done, logits)
                action, logits, hx = model(obs.unsqueeze(0), last_action, reward,
                                           done, hx, actor=True)
                obs, reward, done = step(conn, action)
                total_reward += reward
                last_action = torch.tensor(action, dtype=torch.int64).view(1, 1)
                reward = torch.tensor(reward, dtype=torch.float32).view(1, 1)
                done = torch.tensor(done, dtype=torch.bool).view(1, 1)
                rollout.append(obs, last_action, reward, done, logits.detach())
                steps += 1
    close(conn)
