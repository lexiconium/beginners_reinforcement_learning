from collections import deque, namedtuple
import random

import torch

Transition = namedtuple(
    typename='Transition',
    field_names=('state', 'action', 'reward', 'next_state')
)

def make_batch_dim(x):
    x = torch.tensor(x)
    return x if x.ndim > 1 else x.view(-1, 1)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*map(make_batch_dim, args)))

    def sample(self, batch_size):
        sampled = random.sample(self.memory, k=batch_size)
        sampled = Transition(*zip(*sampled))
        return map(torch.cat, sampled)

    def __len__(self):
        return len(self.memory)