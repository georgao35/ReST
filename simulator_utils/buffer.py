import torch


class Buffer:
    def __init__(self, obs_dim, act_dim, num_envs, traj_len, gamma=0.99, lam=0.95, device='cuda:0'):
        self.device = device
        self.obs_buf = torch.zeros((num_envs, traj_len, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((num_envs, traj_len, act_dim), dtype=torch.float32, device=device)
        self.mu_buf = torch.zeros((num_envs, traj_len, act_dim), dtype=torch.float32, device=device)
        self.std_buf = torch.zeros((num_envs, traj_len, act_dim), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros((num_envs, traj_len), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((num_envs, traj_len), dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros((num_envs, traj_len), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((num_envs, traj_len), dtype=torch.float32, device=device)
        self.logp_buf = torch.zeros((num_envs, traj_len), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros((num_envs, traj_len), dtype=torch.bool, device=device)

        self.gamma, self.lam = gamma, lam
        self.traj_len, self.num_envs = traj_len, num_envs

        self.ptr = 0
        self.path_start_idx = torch.zeros((num_envs), dtype=torch.float32, device=self.device)

    def clear(self):
        self.ptr = 0
        self.path_start_idx = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros((self.num_envs, self.traj_len), dtype=torch.bool, device=self.device)

    def store(self, obs, act, mu, std, rew, val, logp, done):
        assert self.ptr < self.traj_len  # buffer has to have room so you can store
        self.obs_buf[:, self.ptr, :] = obs
        self.act_buf[:, self.ptr, :] = act
        self.mu_buf[:, self.ptr, :] = mu
        self.std_buf[:, self.ptr, :] = std
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
        self.logp_buf[:, self.ptr] = logp
        self.done_buf[:, self.ptr] = done
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        discounted = torch.zeros((x.shape[0], x.shape[1] + 1), dtype=torch.float32, device=self.device)
        for i in reversed(range(discounted.shape[1] - 1)):
            discounted[:, i] = x[:, i] + discount * discounted[:, i + 1] * (~self.done_buf[:, i])
        return discounted[:, :-1]

    def finish_path(self, last_val=0):
        self.rew_buf[:, self.ptr - 1] += self.gamma * last_val * (~self.done_buf[:, self.ptr - 1])
        self.path_start_idx = self.ptr

    def compute_adv(self):
        zero = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        vals = torch.cat((self.val_buf, zero), dim=1)
        qfun = self.rew_buf + (~self.done_buf) * self.gamma * vals[:, 1:]
        delta = qfun - vals[:, :-1]
        self.adv_buf = self.discount_cumsum(delta, self.gamma * self.lam)
        self.ret_buf = self.discount_cumsum(self.rew_buf, self.gamma)

    def get(self):
        assert self.ptr == self.traj_len

        obs = self.obs_buf.view(self.traj_len * self.num_envs, -1)
        act = self.act_buf.view(self.traj_len * self.num_envs, -1)
        mu = self.mu_buf.view(self.traj_len * self.num_envs, -1)
        std = self.std_buf.view(self.traj_len * self.num_envs, -1)
        adv = self.adv_buf.reshape(self.traj_len * self.num_envs)
        ret = self.ret_buf.reshape(self.traj_len * self.num_envs)
        logp = self.logp_buf.view(self.traj_len * self.num_envs)

        adv_mean, adv_std = torch.mean(adv), torch.std(adv)
        adv = (adv - adv_mean) / adv_std

        data = dict(obs=obs,
                    act=act,
                    ret=ret,
                    adv=adv,
                    logp=logp,
                    mu=mu,
                    std=std)
        return {k: v for k, v in data.items()}
