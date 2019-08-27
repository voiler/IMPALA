from functools import partial

import torch
import torch.multiprocessing as mp

from env import Env


class ParameterServer(object):
    def __init__(self):
        self.weight = None

    def pull(self):
        return self.weight

    def push(self, weigth):
        self.weight = weigth


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(conn, env):
    env = env.x()
    while True:
        command, arg = conn.recv()
        if command == 0:
            obs = env.reset()
            conn.send(obs)
        elif command == 1:
            obs, reward, terminal = env.step(arg)
            conn.send([obs, reward, terminal])
        elif command == 2:
            break
        else:
            print("bad command: {}".format(command))
    env.close()
    conn.close()


def make_time_major(batch):
    states = []
    actions = []
    rewards = []
    dones = []
    hx = []
    logits = []
    for t in batch:
        states.append(t.states)
        rewards.append(t.rewards)
        dones.append(t.dones)
        actions.append(t.actions)
        logits.append(t.logit)
        hx.append(t.lstm_hx)
    states = torch.stack(states).transpose(0, 1)
    actions = torch.stack(actions).transpose(0, 1)
    rewards = torch.stack(rewards).transpose(0, 1)
    dones = torch.stack(dones).transpose(0, 1)
    logits = torch.stack(logits).permute(1, 2, 0)
    hx = torch.stack(hx).transpose(0, 1)
    return logits, states, actions, rewards, dones, hx


def combine_time_batch(x, last_action, reward, actor=False):
    if actor:
        return 1, 1, x, last_action, reward
    seq_len = x.shape[0]
    bs = x.shape[1]
    x = x.reshape(seq_len * bs, *x.shape[2:])
    last_action = last_action.reshape(seq_len * bs, -1)
    reward = reward.reshape(seq_len * bs, 1)
    return seq_len, bs, x, last_action, reward


def get_action_size(env_args):
    env = Env(**env_args)
    action_size = env.action_space()
    env.close()
    del env
    return action_size


def env_process(env_args):
    env = partial(Env, **env_args)
    conn, child_conn = mp.Pipe()
    proc = mp.Process(target=worker, args=(child_conn, CloudpickleWrapper(env)))
    return proc, conn
