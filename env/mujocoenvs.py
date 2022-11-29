import gym
import torch
from multiprocessing import Process, Pipe
from tqdm import tqdm

def step(env: gym.Env, conn):
    while True:
        info = conn.recv()
        action = info[0]
        reset = info[1]
        close = info[2]
        if close == True:
            env.close()
            break
        if reset == True:
            o = env.reset()
            conn.send([o])
        else:
            o, r, d, _ = env.step(action)
            conn.send([o, r, d])

class VecMuJoCoEnv():
    def __init__(self, num_envs, env_name) -> None:
        self.envs = []
        self.parent_conn_list = []
        self.child_conn_list = []
        self.process_list = []
        self.num_envs = num_envs
        for i in range(self.num_envs):
            env = gym.make(env_name, exclude_current_positions_from_observation=False)
            self.envs.append(env)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        for i in range(self.num_envs):
            parent_conn, child_conn = Pipe(duplex=True)
            p = Process(target=step, args=(self.envs[i], child_conn))
            self.parent_conn_list.append(parent_conn)
            self.child_conn_list.append(child_conn)
            self.process_list.append(p)
            p.start()
    def reset(self, env_ids=None):
        if env_ids == None:
            env_ids = range(self.num_envs)
        observations = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32, device='cuda:0')
        for i in env_ids:
            self.parent_conn_list[i].send([0, True, False])
        for i in env_ids:
            obs = self.parent_conn_list[i].recv()
            obs = obs[0]
            obs = torch.from_numpy(obs).to(device='cuda:0')
            observations[i, :] = obs
        return observations
    def step(self, action):
        observations = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32, device='cuda:0')
        reward = torch.zeros((self.num_envs), dtype=torch.float32, device='cuda:0')
        done = torch.zeros((self.num_envs), dtype=torch.bool, device='cuda:0')
        for i in range(self.num_envs):
            self.parent_conn_list[i].send([action[i, :].cpu().numpy(), False, False])
        for i in range(self.num_envs):
            rec = self.parent_conn_list[i].recv()
            o = rec[0]
            r = rec[1]
            d = rec[2]
            o = torch.from_numpy(o).to(device='cuda:0')
            observations[i, :] = o
            reward[i] = r
            done[i] = d
        return observations, reward, done, {}
    def close(self):
        for i in range(self.num_envs):
            self.parent_conn_list[i].send([0, True, True])
        for i in range(self.num_envs):
            self.process_list[i].join()
    def get_dims(self):
        return self.obs_dim, self.act_dim

# num_envs = 10
# env = VecMuJoCoEnv(num_envs, 'Ant-v3')
# env.reset()
# o, adim = env.get_dims()
# for i in tqdm(range(1000)):
#     o, r, d, _ = env.step(torch.randn((num_envs, adim), dtype=torch.float32, device='cuda:0'))
#     if d.any():
#         env_ids = torch.nonzero(d, as_tuple=True)[0].to(dtype=torch.int32)
#         env.reset(env_ids)
# env.close()