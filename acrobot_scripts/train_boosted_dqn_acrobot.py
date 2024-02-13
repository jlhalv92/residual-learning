import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from src.algorithms.value_based.residual_dqn import ResNetBoostedResidualDQN
from mushroom_rl.core import Logger, Core

from mushroom_rl.core import Agent
from mushroom_rl.rl_utils.parameters import to_parameter
from src.networks import *
from tqdm import trange

from src.agents.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils import TorchUtils
from src.networks_discrete import Q0, Q1, Q2, Q3, Q4, Network, Network_Q1, Network_Q2, Network_Q3, Network_Q4


def experiment(alg,
               n_epochs,
               logging=True,
               network=None,
               model_name="",
               model_name_logging = "",
               n_steps=0,
               n_episodes=0,
               model_dir="",
               prior_path="",
               task=1.,
               boosting=False,
               project="",
               pi_boosting=False):

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

    target_update_frequency = 200
    initial_replay_size = 1000
    train_frequency = 1
    max_replay_size = 100000
    gradient_steps = 1
    exploration_fraction = 0.5
    exploration_final_eps =.01
    tau = 1

    exploration_range= exploration_fraction * n_steps * n_epochs
    explore_old_steps = 0.* n_steps * n_epochs
    batch_size = 200
    n_features = 64
    input_shape = mdp.info.observation_space.shape

    TorchUtils.set_default_device('cuda')

    approximator_params = dict(network=network,
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

    epsilon = LinearParameter(value=1., threshold_value=exploration_final_eps, n=exploration_range)
    epsilon_test = Parameter(value=0.)
    epsilon_random = Parameter(value=1.)
    rhos = []
    if boosting:
        for path in prior_path:
            print(path)
            rhos.append(Agent.load(path))
        pi = EpsGreedy(epsilon=epsilon_random, rhos=rhos, boosting=boosting)
    else:
        pi = EpsGreedy(epsilon=epsilon_random)

    agent = alg(mdp.info, pi, NumpyTorchApproximator, approximator_params=approximator_params, **algorithm_params)
    core = Core(agent, mdp)

    if boosting:
        agent.setup_boosting(rhos)

    pi.set_mode("training")
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size,  render=False)


    for n in trange(n_epochs, leave=False):

        pi.set_epsilon(epsilon)
        pi.set_mode("training")
        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        print(explore_old_steps, pi.counter, pi.counter < explore_old_steps)

        pi.set_mode("evaluation")
        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=10, render=False, quiet=True)


        J = np.mean(dataset.compute_J(mdp.info.gamma))
        R = np.mean(dataset.compute_J())

        Q = np.array(agent.Q).mean()
        old_Q = np.array(agent.prior_Q).mean()
        rho = np.array(agent.rho).mean()
        q_loss = core.agent.approximator[0].loss_fit


        logs_dict = {"RETURN": J, "REWARD": R, "q_loss":q_loss, "Q":Q, "old_Q":old_Q,"rho":rho}

        if logging:
            wandb.log(logs_dict, step=n)
        logger.epoch_info(n, J=J, R=R,q_loss=q_loss, Q=Q, old_Q=old_Q,rho=rho)

    agent.save(model_path)

    if logging:
        wandb.finish()

    # logger.info('Press a button to visualize')
    # input()
    # core.env = Gym(env_name,horizon, gamma,mode='human')
    # core.evaluate(n_episodes=1, render=True)


if __name__ == '__main__':


    seed = 10
    nets = [Q0, Q0, Q0, Q0, Q0, Q0]
    tasks = [0.8, 1., 1.2, 1.8, 2.]

    for i in range(1,seed):
        prior_paths = ["src/checkpoint/Acrobot/BOOSTED/agent_Boosted_task_m={}_kg_{}".format(task, i) for task in
                       tasks]
        for task_id, task in enumerate(tasks):
            if task_id <=3:
                continue
                boosting = False
                prior_path = ""
            else:
                boosting = True
                prior_path = prior_paths[:task_id]

            model_name = "agent_Boosted_task_m={}_kg".format(task)
            experiment(alg=ResNetBoostedResidualDQN,
                       n_epochs=30,
                       network=nets[task_id],
                       logging=True,
                       model_name="{}_{}".format(model_name,i),
                       model_name_logging="{}".format(model_name),
                       model_dir="../src/checkpoint/Acrobot/BOOSTED",
                       n_steps=5000,
                       n_episodes=10,
                       task=task,
                       project="Acrobot_dqn_dashboard_2_task_{}".format(task),
                       prior_path=prior_path,
                       boosting =boosting)
