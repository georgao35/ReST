from simulator_utils.user_config import DEFAULT_DATA_DIR, FORCE_DATESTAMP
import os.path as osp
import time
import torch
import os

DIV_LINE_WIDTH = 80


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.

    If no seed is given and datestamp is false, 

    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,

    ::

        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to

    ::

        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in 
    ``spinup/user_config.py``. 

    Args:

        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:

        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


class RunningStats:
    """
    Calculate normalized input from running mean and std
    See https://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, obs_dim=30, num_envs=4000, device='cuda:0', clip=1e6):
        self.x = torch.zeros((obs_dim), dtype=torch.float32, device=device)  # Current value of data stream
        self.mean = torch.zeros((obs_dim), dtype=torch.float32, device=device)  # Current mean
        self.sumsq = torch.zeros((obs_dim), dtype=torch.float32,
                                 device=device)  # Current sum of squares, used in var/std calculation

        self.var = torch.zeros((obs_dim), dtype=torch.float32, device=device)  # Current variance
        self.std = torch.ones((obs_dim), dtype=torch.float32, device=device)  # Current std

        self.count = 0  # Counter

        self.clip = clip

        self.minimum = torch.ones((obs_dim), dtype=torch.float32, device=device) * 1e-2

        self.num_envs = num_envs

    def save_model(self, root):
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(self.x, os.path.join(root, 'x.pt'))
        torch.save(self.mean, os.path.join(root, 'mean.pt'))
        torch.save(self.sumsq, os.path.join(root, 'sumsq.pt'))
        torch.save(self.var, os.path.join(root, 'var.pt'))
        torch.save(self.std, os.path.join(root, 'std.pt'))
        torch.save(self.count, os.path.join(root, 'count.pt'))

    def load_model(self, root):
        self.x = torch.load(os.path.join(root, 'x.pt'))
        self.mean = torch.load(os.path.join(root, 'mean.pt'))
        self.sumsq = torch.load(os.path.join(root, 'sumsq.pt'))
        self.var = torch.load(os.path.join(root, 'var.pt'))
        self.std = torch.load(os.path.join(root, 'std.pt'))
        self.count = torch.load(os.path.join(root, 'count.pt'))

    def push(
            self,
            x: torch.Tensor
    ):
        self.x = x
        self.count += x.shape[0]
        if self.count == self.num_envs:
            self.mean = x.mean(dim=0)
            self.var = x.var(dim=0)
            self.std = x.std(dim=0)
            self.std = torch.maximum(self.std, self.minimum)
        else:
            old_mean = self.mean.clone()
            self.mean += (x - self.mean).sum(dim=0) / self.count

            self.sumsq += ((x - old_mean) * (x - self.mean)).sum(dim=0)

            self.var = self.sumsq / (self.count - 1)

            self.std = torch.sqrt(self.var)
            self.std = torch.maximum(self.std, self.minimum)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def get_std(self):
        return self.std

    def normalize(self, x=None):
        if x is not None:
            self.push(x)
            if self.count <= self.num_envs and self.num_envs == 1:
                return self.x
            else:
                output = (self.x - self.mean) / self.std
        else:
            output = (self.x - self.mean) / self.std
        return torch.clamp(output, -self.clip, self.clip)

    def normalize_wo_push(self, x):
        output = (x - self.mean) / self.std
        # print(output)
        return torch.clamp(output, -self.clip, self.clip)
