"""Learner with parameter server"""

import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import vtrace
from utils import make_time_major


class Learner(mp.Process):
    """Learner to get trajectories from Actors."""

    def __init__(self, model, data, gamma, batch_size, baseline_cost, entropy_cost, lr, decay, momentum, epsilon):
        super().__init__()
        self.model = model
        self.data = data
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, eps=epsilon, weight_decay=decay,
                                       momentum=momentum)
        self.batch_size = batch_size
        self.baseline_cost = baseline_cost
        self.entropy_cost = entropy_cost
        self.gamma = gamma

    def run(self):
        """Gets trajectories from actors and trains learner."""
        batch = []
        while True:
            trajectory = self.data.get()
            # trajectory.cuda()
            batch.append(trajectory)
            if len(batch) < self.batch_size:
                continue
            logits, states, actions, rewards, dones, hx = make_time_major(batch)
            self.optimizer.zero_grad()
            loss = self.train(logits, states, actions, rewards, dones, hx)
            loss.backward()
            self.optimizer.step()
            batch = []

    def train(self, behaviour_logits, states, actions, rewards, dones, hx):
        logits, values = self.model(states, actions, rewards, dones, hx=hx)
        bootstrap_value = values[-1]
        discounts = (~dones).float() * self.gamma
        vs, pg_advantages = vtrace.from_logits(
            behaviour_policy_logits=behaviour_logits,
            target_policy_logits=logits,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value)
        # policy gradient loss
        cross_entropy = F.cross_entropy(logits, actions, reduction='none')
        loss = (cross_entropy * pg_advantages.detach()).sum()
        # baseline_loss
        loss += self.baseline_cost * .5 * (vs - values).pow(2).sum()
        # entropy_loss
        loss += self.entropy_cost * -(-F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(-1).sum()
        return loss
