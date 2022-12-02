# Unsupervised Skill Discovery via Recurrent Skill Training (ReST)

 [**Project**](https://sites.google.com/view/neurips22-rest/home) | [**Paper**]() 

We propose a new way to discover state-covering and dynamic skills without external supervision. Our methods recurrently trains skills to be different from others, and achieved promising results in both 2D mazes and MuJoCo locomotion tasks. 

## Get started

### Clone this repo

```bash
git clone https://github.com/georgao35/ReST.git
cd ReST/
```

### Environment setup

- Install MuJoCo version 2.0 packages from [here](https://www.roboti.us/download.html). You may use the [free license](https://www.roboti.us/license.html) provided.

- Install other dependencies

  `pip install -r requirements.txt`

### Training

#### training command

```bash
# train half cheetah
python main.py --env HalfCheetah --expname halfcheetah --epochs 50 --num_skills 10
# train hopper
python main.py --env Hopper --expname hopper --epochs 50 --num_skills 10
# train walker
python main.py --env Walker2d --expname walker --epochs 50 --num_skills 10
```

The training process is set on GPU (CUDA:0) by default. To specify the GPU, it's recommended to set 
`CUDA_VIISBLE_DEVICES` variables. To use cpu, replace all `'cuda:0'` in codes to `'cpu'`

Training scripts is in `scripts/train.sh`.

#### training configuration

The hyper parameters of the algorithms and environments are set in file `config.yaml`.

### Rendering results

The trained models are saved to subdirectory `rnd_models_single_run/`. 
They are stored in the following pattern: `rnd_models_single_run/<timestep>/<epoch>/<skid>/19`, with extra subdirectories
that store the weights of each skill's policy.

Here are two python files that will report the same metrics and plots as those in papers. They are by default running on 
`'cuda:0'`, and you can change it as described in the training session.

#### Rendering MuJoCo locomotion skills

usage: `python render_mujoco.py`

- `--env`: the mujoco task used. same as `--env` in training part.
- `--timestamp`: the `<timestep>` of the experiment in training.
- `--epoch`: the `<epoch>` trained in training and storage path.
- `--skid`: the `<skid>`(iteration of skills) within each recurrent cycle in the storage path.
- a full description can be found by `python render_mujoco.py -h`.

The videos will be stored in `videos/`, with subdirectories represents different skills.
The qualitative results in the paper will be saved as pdf in `renders/`

## Results

The gifs of our learned policy can be found at the [project website](https://sites.google.com/view/neurips22-rest/home)

## Reference

If you found our work or this code implementation helpful, please cite this work:

```

```

