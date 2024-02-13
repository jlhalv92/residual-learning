import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from src.algorithms.value_based.dqn import DQN
from mushroom_rl.core import Logger, Core

from mushroom_rl.core import Agent
from mushroom_rl.rl_utils.parameters import to_parameter
from src.networks import *
from tqdm import trange
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.rl_utils.parameters import LinearParameter, Parameter
from mushroom_rl.approximators.parametric import NumpyTorchApproximator
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.environments.car_on_hill import CarOnHill
from mushroom_rl.utils import TorchUtils

class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action=None):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted

def experiment(alg,
               n_epochs,
               logging=True,
               model_name="",
               model_name_logging = "",
               n_steps=0,
               n_episodes=0,
               model_dir="",
               prior_path=""):

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
            project="benchmark_Acrobot",
            name=exp_name
        )
    # MDP
    # Parameters
    horizon = 500
    gamma = 0.99
    env_name = 'Acrobot-v1'
    mdp = Gym(env_name,horizon, gamma)
    mdp.env.env.env.env.LINK_MASS_2 = 1.5

    # target_update_frequency = 200
    # initial_replay_size = 500
    # train_frequency = 5
    # max_replay_size = 50000
    # gradient_steps = 1
    # exploration_fraction = 0.2
    # exploration_final_eps =.01
    # # n= exploration_fraction*n_steps*n_epochs
    # n=exploration_fraction*n_steps*n_epochs
    # batch_size = 200
    # n_features = 64


    target_update_frequency = 200
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
    approximator_params = dict(network=Network,
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

    epsilon = LinearParameter(value=1., threshold_value=exploration_final_eps, n=n)
    epsilon_test = Parameter(value=0.)
    epsilon_random = Parameter(value=1.)

    pi = EpsGreedy(epsilon=epsilon_random)
    agent = alg(mdp.info, pi, NumpyTorchApproximator, approximator_params=approximator_params, **algorithm_params)
    core = Core(agent, mdp)

    transfer_agent = Agent.load(prior_path)
    agent.load_transfer(transfer_agent)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size,  render=False)

    pi.set_epsilon(epsilon_test)
    dataset = core.evaluate(n_episodes=n_episodes, render=False, quiet=False)

    J = np.mean(dataset.compute_J(mdp.info.gamma))
    R = np.mean(dataset.compute_J())

    logger.epoch_info(0, J=J, R=R)


    for n in trange(n_epochs, leave=False):

        pi.set_epsilon(epsilon)

        core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
        print(epsilon.get_value())

        pi.set_epsilon(epsilon_test)
        dataset = core.evaluate(n_episodes=n_episodes, render=False, quiet=True)

        J = np.mean(dataset.compute_J(mdp.info.gamma))
        R = np.mean(dataset.compute_J())

        Q = np.array(agent.Q).mean()
        q_loss = core.agent.approximator[0].loss_fit


        logs_dict = {"RETURN": J, "REWARD": R, "q_loss":q_loss, "Q":Q}

        if logging:
            wandb.log(logs_dict, step=n)
        logger.epoch_info(n, J=J, R=R,q_loss=q_loss, Q=Q)

    agent.save(model_path)

    if logging:
        wandb.finish()

    # logger.info('Press a button to visualize')
    # input()
    # core.env = Gym(env_name,horizon, gamma,mode='human')
    # core.evaluate(n_episodes=1, render=True)


if __name__ == '__main__':

    experiments = [1]



    seed = 5
    for i in range(seed):

        model_name = "agent_transfer_dqn_Acrobot"
        experiment(alg=DQN,
                   n_epochs=30,
                   logging=False,
                   model_name="{}_{}".format(model_name,i),
                   model_name_logging="{}".format(model_name),
                   model_dir="../src/checkpoint/Acrobot",
                   n_steps=5000,
                   n_episodes=10,
                   prior_path="src/checkpoint/Acrobot/agent_dqn_Acrobot_0")

