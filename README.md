# Unsupervised Skill Discovery via Recurrent Skill Training (ReST)

 [**Project**](https://sites.google.com/view/neurips22-rest/home) | [**Paper**](https://arxiv.org/) 

**notes: We are working hard to provide you with codes of high reproducibility and easy use. We will release full source code by November.**

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

`python main.py`

Parameters:

- `--env`: our method can be trained on 2D navigation (parameter value: `nav2d`), and MuJoCo tasks including Hopper (`hopper`), Walker2d (`walker`), and HalfCheetah(`halfcheetah`).

- `--algo`: use PPO here.

- `--epochs`: number of training epochs.

- `--save_dir`: the directory to save the progress of experiments.

- `--num_skills`: the number of skills to be discovered.

- `--expname`: the type of the experiment done in the environment. 

  For `nav2d`, it can be chosen from `center`, `door`, `room`, `quarter`, corresponding to the mazes in paper. For MuJoCo tasks, it's the same as the `--env` parameter.

For command usage examples, please refer to our example training script `scripts/run_examples.sh`.

#### training configuration

The hyper parameters of the algorithms and environments are set in file `config.yaml`.

### Rendering results

Our trained models can be achieved here (We are finding a way to make the model more accessible). Download the one you need and extract it to `model/` relative to the root directory. You can also use the model you trained, which can be also found in `model/`.

#### Rendering 2D navigation skills

`python render_nav.py`

Command line parameters:

- `--env`: the type of navigation maze used. same as `--exp_name` in training part.
- `--ts`: the timestep of the experiment in training.
- `--epoch`: the epoch trained in training.
- `--skid`: the iteration of skills within each recurrent cycle.

Results will be stored in `fig/render.pdf`.

#### Rendering MuJoCo locomotion skills

`python render_mujoco.py`

Command line parameters are same as above. 

The videos will be stored in `renders`, with  subdirectories represents different skills.

## Results

|              | skills          |
| ------------ | --------------- |
| Walker 2d    | /to be uploaded |
| Hopper       |                 |
| Half Cheetah |                 |
| Humanoid     |                 |

## Reference

If you found our work or this code implementation helpful, please cite this work:

```

```

