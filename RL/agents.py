import os
import io
import re
from collections import namedtuple
from collections import deque
import torch
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from models import Simple_Net

Observations = namedtuple("Observations", ["s", "a", "r", "n_s", "d"])

class ReplayMemory(object):
    def __init__(self, buffer_len):
        self.memory = deque(maxlen=buffer_len)
    
    def __len__(self):
        return len(self.memory)

    def push(self, observations):
        self.memory.append(observations)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class Simple_DQNAgent:
    def __init__(self, num_states, num_actions, batch_size, gamma, device):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.model = Simple_Net(self.num_states, self.num_actions)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = ReplayMemory(10000)

    def train_loop(self):

        if len(self.memory) < self.batch_size:
            return
        
        observations = self.memory.sample(self.batch_size)

        batch = Observations(*zip(*observations))

        n_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.n_s)))

        s_batch = torch.cat(batch.s)
        a_batch = torch.cat(batch.a)
        r_batch = torch.cat(batch.r)
        n_n_s_batch = torch.cat([s for s in batch.n_s if s is not None])

        self.model.eval()

        q = self.model(s_batch).gather(1, a_batch)
        n_s_v = torch.zeros(self.batch_size).type(torch.FloatTensor)
        n_s_v[n_final_mask] = self.model(n_n_s_batch).data.max(1)[0]
        expected_v = r_batch + self.gamma * n_s_v

        self.model.train()

        loss = F.smooth_l1_loss(q, expected_v)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def policy(self, s, episode):
        epsilon = 0.5 / (episode + 1)

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()

            action = self.model(s).data.max(1)[1].view(1, 1)
        
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action

    def memorize(self, s, a, n_s, r):

        self.memory.push(Observations(s, a, n_s, r))
    
