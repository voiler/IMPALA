"""Learner with parameter server"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import vtrace
from utils import make_time_major


def learner(model, data, ps, args):
    """Learner to get trajectories from Actors."""
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=args.epsilon,
                              weight_decay=args.decay,
                              momentum=args.momentum)
    batch_size = args.batch_size
    baseline_cost = args.baseline_cost
    entropy_cost = args.entropy_cost
    gamma = args.gamma
    save_path = args.save_path
    """Gets trajectories from actors and trains learner."""
    batch = []
    best = 0.
    while True:
        trajectory = data.get()
        batch.append(trajectory)
        if torch.cuda.is_available():
            trajectory.cuda()
        if len(batch) < batch_size:
            continue
        behaviour_logits, obs, actions, rewards, dones, hx = make_time_major(batch)
        optimizer.zero_grad()
        logits, values = model(obs, actions, rewards, dones, hx=hx)
        bootstrap_value = values[-1]
        actions, behaviour_logits, dones, rewards = actions[1:], behaviour_logits[1:], dones[1:], rewards[1:]
        logits, values = logits[:-1], values[:-1]
        discounts = (~dones).float() * gamma
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
        loss += baseline_cost * .5 * (vs - values).pow(2).sum()
        # entropy_loss
        loss += entropy_cost * -(-F.softmax(logits, 1) * F.log_softmax(logits, 1)).sum(-1).sum()
        loss.backward()
        optimizer.step()
        model.cpu()
        ps.push(model.state_dict())
        if rewards.mean().item() > best:
            torch.save(model.state_dict(), save_path)
        if torch.cuda.is_available():
            model.cuda()
        batch = []
