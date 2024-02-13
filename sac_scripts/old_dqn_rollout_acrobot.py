import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from src.algorithms.value_based.residual_dqn import BoostedDQN
from mushroom_rl.core import Logger, Core

from mushroom_rl.core import Agent
from mushroom_rl.rl_utils.parameters import to_parameter
from src.networks import *
from tqdm import trange

from src.agents.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.utils import TorchUtils
from src.networks_discrete import Q0, Q1, Network


def experiment(alg,
               n_epochs,
               logging=True,
               net=None,
               model_name="",
               model_name_logging = "",
               n_steps=0,
               n_episodes=0,
               model_dir="",
               prior_path="",
               task=1.,
               project="",
               explore_old_pi=False):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)


    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, model_name)

    # Algorithm

    exp_name = model_name_logging

    if logging:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            name=exp_name
        )
    # MDP
    # Parameters
    horizon = 500
    gamma = 0.99
    env_name = 'Acrobot-v1'
    mdp = Gym(env_name,horizon, gamma)
    mdp.env.env.env.env.LINK_MASS_2 = task

    target_update_frequency = 100
    initial_replay_size = 1000
    train_frequency = 1
    max_replay_size = 100000
    gradient_steps = 1
    exploration_fraction = 0.5
    exploration_final_eps =.01
    tau = 1

    n=exploration_fraction*n_steps*n_epochs
    batch_size = 200
    n_features = 64
    input_shape = mdp.info.observation_space.shape

    TorchUtils.set_default_device('cuda')
    approximator_params = dict(network=net,
                               optimizer={'class': optim.Adam,
                                          'params': {'lr': .001}},
                               loss=F.smooth_l1_loss,
                               n_features=n_features,
                               input_shape=input_shape,
                               output_shape=mdp.info.action_space.size,
                               n_actions=mdp.info.action_space.n,
                               use_cuda=True
                               )



    algorithm_params = dict(
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        gradient_steps=gradient_steps,
        tau=tau
    )

    epsilon_test = Parameter(value=0.)

    agent = Agent.load(prior_path)

    pi = EpsGreedy(epsilon=epsilon_test)

    core = Core(agent, mdp)



    for n in trange(n_epochs, leave=False):

        pi.set_mode("evaluation")
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=n_episodes, render=False, quiet=True)

        J = np.mean(dataset.compute_J(mdp.info.gamma))
        R = np.mean(dataset.compute_J())

        Q = 0
        old_Q = 0
        rho = 0
        q_loss = 0


        logs_dict = {"RETURN": J, "REWARD": R, "q_loss":q_loss, "Q":Q, "old_Q":old_Q,"rho":rho}

        if logging:
            wandb.log(logs_dict, step=n)
        logger.epoch_info(n, J=J, R=R,q_loss=q_loss, Q=Q, old_Q=old_Q,rho=rho)



    if logging:
        wandb.finish()

    # logger.info('Press a button to visualize')
    # input()
    # core.env = Gym(env_name,horizon, gamma,mode='human')
    # core.evaluate(n_episodes=1, render=True)


if __name__ == '__main__':


    seed = 10
    for task in [1.5, 2., 3.]:
        for i in range(seed):
            model_name = "agent_nominal_dqn_task={}_rollout".format(task)
            experiment(alg=BoostedDQN,
                       n_epochs=30,
                       net=Q0,
                       logging=True,
                       model_name="{}_{}".format(model_name,i),
                       model_name_logging="{}".format(model_name),
                       model_dir="../src/checkpoint/Acrobot",
                       n_steps=5000,
                       n_episodes=10,
                       prior_path="../src/checkpoint/Acrobot/Nominal/Agent_res_dqn_rho_0_0",
                       task=task,
                       project="Acrobot_dqn_task_{}".format(task),
                       explore_old_pi=True)


