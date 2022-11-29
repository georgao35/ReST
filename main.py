from simulator_utils.simulator import Simulator
from simulator_utils.run_utils import setup_logger_kwargs
from simulator_utils.logx import EpochLogger
import argparse
from trainer_utils.collection import MlpActorCriticCollection
from datetime import datetime
import os
import yaml

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


def train(env_name, epochs, algo, save_dir, num_skills, expname):

    exp_name = expname
    data_dir = os.path.join(save_dir, env_name, TIMESTAMP)
    logger_kwargs = setup_logger_kwargs(exp_name, 0, data_dir=data_dir)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    file = open('config.yaml', 'r', encoding='utf-8')
    f = file.read()
    config = yaml.load(f, Loader=yaml.FullLoader)

    simulator = Simulator(config[env_name], expname)

    obs_dim, act_dim, exclude = simulator.get_info()

    trainer_collection = MlpActorCriticCollection(env_name, algo, config[algo],
                                                  obs_dim, act_dim,
                                                  config[env_name]['num_envs'],
                                                  num_skills, exclude)

    for t in range(3):
        data, _, _ = simulator.run_sim(trainer_collection.trainer_list[0].agent,
                                       trainer_collection)
        simulator.clear()
        trainer_collection.trainer_list[0]._update_rnd(data)
    trainer_collection.activate(0)
    env_interacts = 0
    for epoch in range(epochs):

        for skill in range(num_skills):
            if epoch == 0 and skill == 0:
                continue
            trainer_collection.deactivate(skill)
            for t in range(20):
                data, ep_ret_list, ep_len_list = simulator.run_sim(
                    trainer_collection.trainer_list[skill].agent,
                    trainer_collection)
                for ep_ret in ep_ret_list:
                    logger.store(CumulativeReward=ep_ret)
                for ep_len in ep_len_list:
                    logger.store(EpisodeLength=ep_len)
                simulator.clear()
                train_info = trainer_collection.trainer_list[
                    skill].update_params(data, t)
                env_interacts += config[env_name]['num_envs'] * config[
                    env_name]['traj_len']
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('Skill', skill)
                logger.log_tabular('CumulativeReward', with_min_and_max=True)
                logger.log_tabular('EpisodeLength', with_min_and_max=True)
                logger.log_tabular('LossPi', train_info['loss_pi'])
                logger.log_tabular('LossValue', train_info['loss_v'])
                logger.log_tabular('LossRND', train_info['loss_rnd'])
                logger.log_tabular('TotalEnvInteracts', env_interacts)
                logger.dump_tabular()
            if not os.path.exists('./rnd_models_single_run'):
                os.mkdir('rnd_models_single_run')
            if not os.path.exists(os.path.join('./rnd_models_single_run', TIMESTAMP)):
                os.mkdir(os.path.join('./rnd_models_single_run', TIMESTAMP))
            if not os.path.exists(os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch))):
                os.mkdir(os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch)))
            if not os.path.exists(os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch), str(skill))):
                os.mkdir(os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch), str(skill)))
            os.mkdir(os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch), str(skill), str(t)))
            trainer_collection.save_model(
                os.path.join('./rnd_models_single_run', TIMESTAMP, str(epoch), str(skill), str(t)))
            if epoch == 0:
                trainer_collection.activate(skill)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='nav2d', help='type of environment')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument('--save_dir', type=str, default='./data/', help='directory that stores training progress')
    parser.add_argument('--num_skills', type=int, default=10, help='number of skills to be discovered')
    parser.add_argument('--expname', type=str, default='exp', help='type of the experiment')
    args = parser.parse_args()

    train(env_name=args.env,
          epochs=args.epochs,
          algo=args.algo,
          save_dir=args.save_dir,
          num_skills=args.num_skills, 
          expname=args.expname)
