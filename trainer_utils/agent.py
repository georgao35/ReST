import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Independent
from simulator_utils.run_utils import RunningStats
from torch.nn import functional as F


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MlpActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_sizes, activation,
                 output_activation, log_std_init, independent) -> None:
        super(MlpActor, self).__init__()
        log_std = log_std_init * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hid_sizes) + [act_dim], activation,
                          output_activation)
        self.independent = independent

    def forward(self, obs):
        mu = self.mu_net(obs)

        std = torch.exp(self.log_std)
        if self.independent:
            pi = Independent(Normal(mu, std), 1)
        else:
            pi = Normal(mu, std)
        a = pi.sample()
        return a, mu, std, pi


class MlpRND(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_sizes, activation) -> None:
        super(MlpRND, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.hid_sizes = hid_sizes
        self.target_net = MlpNetwork(obs_dim, hid_sizes, activation)
        self.source_net = MlpNetwork(obs_dim, hid_sizes, activation)
        self.criterion = nn.MSELoss()
        for param in self.target_net.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        target = self.target_net(obs)
        source = self.source_net(obs)
        return self.criterion(source, target)

    def get_rnd_reward(self, obs: torch.Tensor) -> torch.Tensor:
        target = self.target_net(obs)
        source = self.source_net(obs)
        return torch.mean((target - source) ** 2, dim=1)

    def reset(self) -> None:
        self.target_net = MlpCritic(self.obs_dim, self.hid_sizes, self.activation).to(device='cuda:0')
        self.source_net = MlpCritic(self.obs_dim, self.hid_sizes, self.activation).to(device='cuda:0')
        for param in self.target_net.parameters():
            param.requires_grad = False


class MlpCritic(nn.Module):
    def __init__(self, obs_dim, hid_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hid_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs),
                             -1)  # Critical to ensure v has right shape.


class MlpNetwork(nn.Module):
    def __init__(self, obs_dim, hid_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hid_sizes) + [5], activation)

    def forward(self, obs):
        return self.v_net(obs)


class VAEModel(nn.Module):
    def __init__(self, obs_dim, vae_beta=0.5):
        super().__init__()
        self.code_dim = 128

        self.make_networks(obs_dim, self.code_dim)
        self.beta = vae_beta

        self.apply(weight_init)

    def make_networks(self, obs_dim, code_dim):
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU()
        )
        self.enc_mu = nn.Linear(150, code_dim)
        self.enc_logvar = nn.Linear(150, code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, obs_dim)
        )

    def encode(self, obs):
        enc_features = self.enc(obs)
        mu = self.enc_mu(enc_features)
        logvar = self.enc_logvar(enc_features)
        stds = (0.5 * logvar).exp()
        return mu, logvar, stds

    def forward(self, obs_z, epsilon):
        mu, logvar, stds = self.encode(obs_z)
        code = epsilon * stds + mu
        obs_distr_params = self.dec(code)
        return obs_distr_params, (mu, logvar, stds)

    def get_loss(self, obs):
        epsilon = torch.randn([obs.shape[0], self.code_dim], dtype=torch.float32, device='cuda:0')
        obs_distr_params, (mu, logvar, stds) = self(obs, epsilon)
        kle = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        log_prob = F.mse_loss(obs, obs_distr_params, reduction='none')

        loss = self.beta * kle + log_prob.mean()
        return loss  # , log_prob.sum(list(range(1, len(log_prob.shape)))).view(log_prob.shape[0], 1)

    def get_reward(self, obs):
        epsilon = torch.zeros([obs.shape[0], self.code_dim], dtype=torch.float32, device='cuda:0')
        obs_distr_params, (mu, logvar, stds) = self(obs, epsilon)
        reward = (obs_distr_params - obs) ** 2
        reward = torch.sum(reward, dim=1)
        return reward


class PenaltyParam(nn.Module):
    def __init__(self, penalty_init):
        super(PenaltyParam, self).__init__()
        penalty_init = float(penalty_init)
        self.penalty = nn.Parameter(torch.log(
            max(torch.exp(torch.tensor([penalty_init])) - 1, 1e-8)),
            requires_grad=True)
        self.activation = nn.Softplus()

    def forward(self, x):
        return x * self.activation(self.penalty[0])

    def pnloss(self, x):
        return x * self.penalty[0]

    def normloss(self, x):
        return x / (1 + self.activation(self.penalty[0]))

    def getpenalty(self):
        return self.activation(self.penalty[0]).detach()


class MlpActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_sizes, activation,
                 output_activation, log_std_init, independent,
                 num_envs, exclude) -> None:
        super(MlpActorCritic, self).__init__()
        self.exclude = exclude
        self.actor = MlpActor(obs_dim - exclude, act_dim, hid_sizes, activation,
                              output_activation, log_std_init, independent)
        self.critic = MlpCritic(obs_dim - exclude, hid_sizes, activation)
        self.normalizer = RunningStats(obs_dim, num_envs)
        self.vae_module = MlpRND(int(obs_dim), act_dim, (512, 512),
                                 activation)
        # VAEModel(int(obs_dim))# MlpRND(int(obs_dim/2), act_dim, hid_sizes, activation)# MODIFIED!!!

    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            obs = self.normalizer.normalize(obs)[:, self.exclude:]

        a, mu, std, pi = self.actor(obs)
        v = self.critic(obs)
        log_p = pi.log_prob(a).sum(axis=-1)
        return a, mu, std, log_p, v, pi

    def get_actor_output(self, obs):
        with torch.no_grad():
            obs = self.normalizer.normalize(obs)[:, self.exclude:]
        return self.actor(obs)

    def get_critic_output(self, obs):
        with torch.no_grad():
            obs = self.normalizer.normalize(obs)[:, self.exclude:]
        return self.critic(obs)

    def get_rnd_module_reward(self, obs):
        return self.vae_module.get_rnd_reward(obs)

    def get_rnd_module_output(self, obs):
        return self.vae_module.forward(obs)

    def step(self, obs):
        with torch.no_grad():
            a, mu, std, log_p, v, _ = self.forward(obs)
            return a, mu, std, log_p, v
