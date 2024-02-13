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


class Q1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(Q1, self).__init__()

        n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        self._h1 = nn.Linear(n_input, n_features)

        self._rho_0 = ResidualBlock(n_features)
        self._rho_1 = ResidualBlock(n_features)
        self._out = LinearOutput(n_features, self.n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self,state, action=None):

        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = self._rho_0(features1)
        q = self._out(features2)

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
               model_dir=""):

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
            project="benchmark_CAR_ON_HILL",
            name=exp_name
        )
    # MDP
    # Parameters
    horizon = 200
    gamma = 0.98
    target_update_frequency = 600
    initial_replay_size = 1000
    train_frequency = 16
    max_replay_size = 10000
    gradient_steps = 8
    exploration_fraction = 0.2
    exploration_final_eps =.07
    mdp = Gym('MountainCar-v0', horizon, gamma)

    optimizer = {'class': optim.Adam, 'params': dict(lr=0.004, eps=1e-8)}

    approximator_params = dict(
        network=Q0,
        input_shape=mdp.info.observation_space.shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=256,
        two_layers=True,
        loss=F.smooth_l1_loss,
        optimizer=optimizer,
        use_cuda=True
    )



    algorithm_params = dict(
        batch_size=128,
        target_update_frequency=target_update_frequency,
        initial_replay_size=initial_replay_size,
        max_replay_size=max_replay_size,
        gradient_steps=gradient_steps
    )

    epsilon = LinearParameter(value=1., threshold_value=exploration_final_eps, n=exploration_fraction*n_steps*n_epochs)
    epsilon_test = Parameter(value=0.)
    epsilon_random = Parameter(value=1.)

    pi = EpsGreedy(epsilon=epsilon_random)
    agent = alg(mdp.info, pi, NumpyTorchApproximator, approximator_params=approximator_params, **algorithm_params)
    core = Core(agent, mdp)


    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

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

    logger.info('Press a button to visualize')
    input()
    core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    experiments = [1]



    seed = 10
    for i in range(seed):

        model_name = "agent_dqn_res"
        experiment(alg=DQN,
                   n_epochs=500,
                   logging=False,
                   model_name="{}_{}".format(model_name,i),
                   model_name_logging="{}".format(model_name),
                   model_dir="../src/checkpoint/car_hill",
                   n_steps= 1200,
                   n_episodes=100)

