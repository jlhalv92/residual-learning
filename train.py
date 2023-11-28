import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# from mushroom_rl.algorithms.actor_critic import SAC
from src.algorithms.actor_critic.sac import SAC
from mushroom_rl.core import Core, Logger
# from mushroom_rl.environments.dm_control_env import DMControl
from src.mushroom_extension.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from mushroom_rl.utils.parameters import to_parameter


from tqdm import trange


class CriticNetwork(nn.Module):
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

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)

        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

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
                                gain=nn.init.calculate_gain('relu'))


    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg, n_epochs, n_steps, n_steps_test, tasks, experiment_name, model,
               load_model=False,
               stop_logging=False,
               model_type="", model_dir=""):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)
    logger.info("Experiment name: " + experiment_name)


    # dir_path =model_dir
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        dir_path = os.path.join(model_dir, model_type)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)


    # MDP
    horizon = 500
    gamma = 0.99

    # Settings
    initial_replay_size = 15000
    max_replay_size = 500000
    batch_size = 256
    n_features = 400
    warmup_transitions = 10000
    tau = 0.01
    lr_alpha = 3e-4
    log_wandb = True

    mdp = DMControl('walker', tasks[0], horizon, gamma, use_pixels=False)

    use_cuda = torch.cuda.is_available()

    if load_model:
        agent_path = os.path.join(model)
        agent = Agent.load(agent_path)

        if model_type == "residual":
            print("activate residual")
            agent._residual = to_parameter(True)

    # Approximator
    else:
        actor_input_shape = mdp.info.observation_space.shape
        actor_mu_params = dict(network=ActorNetwork,
                               n_features=n_features,
                               input_shape=actor_input_shape,
                               output_shape=mdp.info.action_space.shape,
                               use_cuda=use_cuda)
        actor_sigma_params = dict(network=ActorNetwork,
                                  n_features=n_features,
                                  input_shape=actor_input_shape,
                                  output_shape=mdp.info.action_space.shape,
                                  use_cuda=use_cuda)

        actor_optimizer = {'class': optim.Adam,
                           'params': {'lr': 5e-4}}

        critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)

        critic_params = dict(network=CriticNetwork,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': 5e-4}},
                             loss=F.mse_loss,
                             n_features=n_features,
                             input_shape=critic_input_shape,
                             output_shape=(1,),
                             use_cuda=use_cuda)


        # Agent
        agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)
    agent
    # Algorithm
    core = Core(agent, mdp)

    # RUN
    dataset = core.evaluate(n_steps=n_steps_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    exp_name = "walker_SAC_{}_4".format(model_type)


    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="residual_learning_locomotion",
            name=exp_name
        )

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=True)

    for n in trange(n_epochs, leave=False):

        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=True)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n+1, J=J, R=R, entropy=E)
        logs_dict = {"RETURN": J, "REWARD": R, "ENTROPY": E}
        if log_wandb:
            wandb.log(logs_dict)


    agent_path = os.path.join(dir_path, experiment_name + '_{}_{}'.format(tasks[0],model_type))


    agent.save(agent_path)

    if stop_logging:
        wandb.finish()

    # logger.info('Press a button to visualize')
    # input()
    # core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':


    # experiment(alg=SAC,
    #            n_epochs=30,
    #            n_steps=4000,
    #            n_steps_test=3000,
    #            tasks=["run"],
    #            experiment_name="EXP_{}".format(0),
    #            load_model=True,
    #            model="/home/jose-luis/Project/residual_learning/src/checkpoint/nominal_model/model_10",
    #            stop_logging=True,
    #            model_type="walk",
    #            model_dir="src/checkpoint/walker_walk_nominal")

    runs = np.arange(10)
    for run in runs:

        experiment(alg=SAC,
                   n_epochs=80,
                   n_steps=4000,
                   n_steps_test=3000,
                   tasks=["run"],
                   experiment_name="EXP_{}".format(run),
                   load_model=True,
                   model= "/home/jose-luis/Project/residual_learning/src/checkpoint/walker_walk_nominal/walk/EXP_0_walk_walk",
                   stop_logging=True,
                   model_type="residual",
                   model_dir="src/checkpoint/walker_exp_02")
