import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from src.algorithms.actor_critic.brl import BRL
from mushroom_rl.core import Logger, Core
from src.mushroom_extension.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.core import Agent
from mushroom_rl.utils.parameters import to_parameter
from src.networks import *
from tqdm import trange


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
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(alg,
               n_epochs,
               starting_epoch,
               n_steps,
               n_steps_test,
               tasks,
               model=[],
               logging=True,
               model_name="",
               model_dir="",
               boosting=False,
               model_name_logging = "",
               use_kl_on_pi=False,
               hard_task=False,
               target_velocity=8,
               residual_name=None,
               Critic=None):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # dir_path =model_dir

    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker',
                    tasks[0],
                    horizon,
                    gamma,
                    use_pixels=False,
                    hard_task=hard_task,
                    target_velocity=target_velocity)

    # Settings
    initial_replay_size = 15000
    max_replay_size = 500000
    batch_size = 256
    n_features = 400
    warmup_transitions = 10000
    tau = 0.01
    lr_alpha = 3e-4
    log_wandb = logging
    lr = 5e-4
    log_std_min=-3


    use_cuda = torch.cuda.is_available()

    # Approximator

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
                       'params': {'lr': lr}}


    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    # critic_input_shape = (n_features,)

    critic_params = dict(network=Critic,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr':lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, model_name)

    if boosting:

        prior_paths = model
        prior_agents = list()
        for agent_path in prior_paths:
            prior_agents.append(Agent.load(agent_path))

        agent = alg(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer,critic_params,
                     batch_size=batch_size,initial_replay_size=initial_replay_size,
                     max_replay_size=max_replay_size,
                    warmup_transitions=warmup_transitions,
                     tau=tau,
                    lr_alpha=lr_alpha,
                    log_std_min=log_std_min,
                    prior_agent=prior_agents[0],
                    use_entropy=True)

        prior_model = Agent.load(agent_path)
        agent.setup_boosting(prior_agents=prior_agents,
                             prior_model=prior_model,
                             residual_name=residual_name,
                             use_kl_on_pi=use_kl_on_pi,
                             kl_on_pi_alpha=1)
    else:
        agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                    actor_optimizer, critic_params, batch_size, initial_replay_size,
                    max_replay_size, warmup_transitions, tau, lr_alpha,
                    critic_fit_params=None)

    # Algorithm
    core = Core(agent, mdp)
    exp_name = model_name_logging

    if logging:
        wandb.init(
            # set the wandb project where this run will be logged
            project="residual_learning_Resnet_Experiments",
            name=exp_name
        )

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=False)

    dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=False)
    s, *_ = parse_dataset(dataset)


    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    logger.epoch_info(0, J=J, R=R, entropy=E)


    for n in trange(n_epochs, leave=False):

        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        # rho =np.array(core.agent.rho).mean()
        # rho_prior = np.array(core.agent.rho_prior).mean()
        # rho_no_prior = np.array(core.agent.rho_no_prior).mean()
        # core.agent.reset_rho()

        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=False)
        s, *_ = parse_dataset(dataset)
        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        Q = np.array(agent.q).mean()
        old_Q = np.array(agent.q_old).mean()
        rho = np.array(agent.rho).mean()


        q_loss = core.agent._critic_approximator[0].loss_fit

        # core.evaluate(n_episodes=1, render=True, quiet=False)
        # if n%5 == 0:
        #     print("reset")
        #     critic = core.agent._critic_approximator
        #     agent.reset_parameters(critic)
        #     critic = core.agent._target_critic_approximator
        #     agent.reset_parameters(critic)

        logs_dict = {"RETURN": J, "REWARD": R, "ENTROPY": E, "q_loss":q_loss, "Q":Q, "old_Q":old_Q, "rho":rho}

        if logging:
            wandb.log(logs_dict, step=n+starting_epoch)
        logger.epoch_info(n + 1 + starting_epoch, J=J, R=R, entropy=E,q_loss=q_loss, Q=Q, old_Q=old_Q, rho=rho, velocity=np.mean(core.mdp.horizontal_velocity), upright=np.mean(core.mdp.upright))
        core.mdp.horizontal_velocity = []
        core.mdp.upright =[]

    agent.save(model_path)

    if logging:
        wandb.finish()

        # logger.info('Press a button to visualize')
        # input()
        # core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':

    experiments = [2]
    n_steps= 4000 # training
    n_steps_test=3000


    tasks = ["stand", "walk", "run"]
    run_velocities = [0, 1, 8]
    critics = [Q0,Q1,Q0]
    epochs = [20, 20, 100]
    critics = [Q0, Q1, Q0]
    for exp in experiments:

        print("Run experiment {}, for task {} with target velocity {} ". format(exp, tasks[exp], run_velocities[exp]))


        model_name = "agent_run"
        boosting = False
        if exp >= 1:
            boosting = False
        experiment(alg=BRL,
                   n_epochs=epochs[exp],
                   starting_epoch=0,
                   n_steps=n_steps,
                   n_steps_test=n_steps_test,
                   tasks=[tasks[exp]],
                   model=["src/checkpoint/boosted_curriculum/Q_0"],
                   logging=False,
                   model_name="{}_{}".format(model_name,exp),
                   model_name_logging="{}".format(model_name),
                   model_dir="../src/checkpoint/run_walker",
                   boosting=boosting,
                   use_kl_on_pi=False,
                   hard_task=False,
                   target_velocity=run_velocities[exp],
                   residual_name = "_rho_{}".format(exp),
                   Critic=critics[exp]
                   )

