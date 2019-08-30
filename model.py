import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import combine_time_batch


class Network(nn.Module):
    def __init__(self, action_size=16, input_channels=4, hidden_size=512):
        super(Network, self).__init__()
        self.action_space = action_size
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc = nn.Linear(3136, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size + action_size + 1, 256)
        self.head = Head(action_size)

    def forward(self, x, last_action, reward, dones, hx=None, actor=False):
        seq_len, bs, x, last_action, reward = combine_time_batch(x, last_action, reward, actor)
        last_action = torch.zeros(last_action.shape[0], self.action_space,
                                  dtype=torch.float32, device=x.device).scatter_(1, last_action, 1)
        x = F.leaky_relu(self.conv1(x), inplace=True)
        x = F.leaky_relu(self.conv2(x), inplace=True)
        x = F.leaky_relu(self.conv3(x), inplace=True)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc(x), inplace=True)
        x = torch.cat((x, reward, last_action), dim=1)
        x = x.view(seq_len, bs, -1)
        lstm_out = []
        hx = hx.to(x.device)
        init_core_state = torch.zeros((2, bs, 256), dtype=torch.float32, device=x.device)
        for state, d in zip(torch.unbind(x, 0), torch.unbind(dones, 0)):
            hx = torch.where(d.view(1, -1, 1), init_core_state, hx)
            hx = self.lstm(state, hx.unbind(0))
            lstm_out.append(hx[0])
            hx = torch.stack(hx, 0)
        x = torch.cat(lstm_out, 0)
        logits, values = self.head(x, actor)
        logits[torch.isnan(logits)] = 1e-12
        if not actor:
            return logits.view(seq_len, -1, bs), values.view(seq_len, bs)
        else:
            action = torch.softmax(logits, 1).multinomial(1).item()
            return action, logits.view(1, -1), hx


class Head(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.actor_linear = nn.Linear(256, action_space)
        self.critic_linear = nn.Linear(256, 1)

    def forward(self, x, actor):
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values
