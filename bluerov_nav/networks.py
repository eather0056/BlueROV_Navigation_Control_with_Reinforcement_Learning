"""New network implementation; legacy serialized models are not interchangeable."""
import math
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, history=5):
        super().__init__()
        # Kernels/hidden layers are explicit implementation choices, not claimed
        # recovered experimental settings. Feature group sizes follow the chapter.
        self.image = nn.Sequential(
            nn.Conv2d(history, 32, (10, 14), 8, (1, 4)), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 8 * 10, 512), nn.ReLU())
        self.goal = nn.Sequential(nn.Flatten(), nn.Linear(3 * history, 96), nn.ReLU())
        self.range = nn.Sequential(nn.Flatten(), nn.Linear(history, 32), nn.ReLU())
        self.actions = nn.Sequential(nn.Flatten(), nn.Linear(2 * history, 64), nn.ReLU())

    def forward(self, obs):
        return torch.cat((self.image(obs["depth"]), self.goal(obs["goal"]),
                          self.range(obs["range"]), self.actions(obs["actions"])), dim=-1)


class Actor(nn.Module):
    def __init__(self, history=5):
        super().__init__()
        self.encoder = Encoder(history)
        self.head = nn.Sequential(nn.Linear(704, 512), nn.ReLU(), nn.Linear(512, 2))
        self.log_std = nn.Parameter(torch.zeros(2))

    def mean(self, obs):
        return self.head(self.encoder(obs))

    def distribution(self, obs):
        mean = self.mean(obs)
        return Normal(mean, self.log_std.clamp(-10, 2).exp().expand_as(mean))

    def sample(self, obs):
        dist = self.distribution(obs)
        latent = dist.rsample()
        action = latent.tanh()
        # Stable Jacobian; no clipping probability mass at the boundaries.
        log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        log_prob = (dist.log_prob(latent) - log_jacobian).sum(-1, keepdim=True)
        return action, log_prob, latent

    def log_prob(self, obs, latent):
        jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return (self.distribution(obs).log_prob(latent) - jacobian).sum(-1, keepdim=True)

    def deterministic(self, obs):
        return self.mean(obs).tanh()


class Value(nn.Module):
    def __init__(self, history=5):
        super().__init__()
        self.encoder = Encoder(history)
        self.head = nn.Sequential(nn.Linear(704, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, obs):
        # Unbounded critic: shaped return need not lie in [-1, 1].
        return self.head(self.encoder(obs))


class Critic(nn.Module):
    def __init__(self, history=5):
        super().__init__()
        self.encoder = Encoder(history)
        self.head = nn.Sequential(nn.Linear(706, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, obs, action):
        return self.head(torch.cat((self.encoder(obs), action), dim=-1))


class TwinQ(nn.Module):
    def __init__(self, history=5):
        super().__init__()
        self.q1, self.q2 = Critic(history), Critic(history)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
