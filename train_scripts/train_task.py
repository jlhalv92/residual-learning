import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# from mushroom_rl.algorithms.actor_critic import SAC
from src.algorithms.actor_critic.boost_sac import SAC
from src.algorithms.actor_critic.residual_sac import SAC

from mushroom_rl.core import Logger
from src.mushroom_extension.core_noise import Core
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


def experiment(alg,
               n_epochs,
               starting_epoch,
               n_steps,
               n_steps_test,
               tasks,
               model=[],
               logging=False,
               model_name="",
               model_dir="",
               use_prior=False,
               model_name_logging = "",
               use_kl_on_pi=False,
               transfer =False,
               test=False):

    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)



    # dir_path =model_dir
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    model_path = os.path.join(model_dir, model_name)

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
    log_wandb = logging

    mdp = DMControl('walker', tasks[0], horizon, gamma, use_pixels=False)

    if test:
        print("holi")
        agent = Agent.load(model[0])
        core = Core(agent, mdp)
        dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)
        logger.epoch_info(0, J=J, R=R, entropy=E)

    else:
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

        if transfer:
            agent = Agent.load(model[0])
            agent.setup_transfer()

        else:
        # Agent


            agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                        actor_optimizer, critic_params, batch_size, initial_replay_size,
                        max_replay_size, warmup_transitions, tau, lr_alpha,
                        critic_fit_params=None)


            if use_prior:
                prior_paths = model
                prior_agents = list()
                for agent_path in prior_paths:
                    prior_agents.append(Agent.load(agent_path))

                agent.setup_boosting(prior_agents=prior_agents,
                                     use_kl_on_pi=use_kl_on_pi,
                                     kl_on_pi_alpha=1e-3)



        # Algorithm
        core = Core(agent, mdp)
        exp_name = model_name_logging

        if logging:
            wandb.init(
                # set the wandb project where this run will be logged
                project="residual_learning_Exp_mass_20_extra",
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
            rho =np.array(core.agent.rho).mean()
            rho_prior = np.array(core.agent.rho_prior).mean()
            rho_no_prior = np.array(core.agent.rho_no_prior).mean()
            core.agent.reset_rho()

            dataset = core.evaluate(n_steps=n_steps_test, render=False, quiet=False)
            s, *_ = parse_dataset(dataset)
            J = np.mean(compute_J(dataset, mdp.info.gamma))
            R = np.mean(compute_J(dataset))
            E = agent.policy.entropy(s)
            q_loss = core.agent._critic_approximator[0].loss_fit
            # core.evaluate(n_episodes=1, render=True, quiet=False)

            if use_prior:
                logs_dict = {"RETURN": J, "REWARD": R, "ENTROPY": E, "Q":rho, "rho_prior":rho_prior, "rho":rho_no_prior ,"q_loss":q_loss}
            else:
                logs_dict = {"RETURN": J, "REWARD": R, "ENTROPY": E, "Q": rho, "q_loss":q_loss}

            if logging:
                wandb.log(logs_dict, step=n+starting_epoch)
            logger.epoch_info(n + 1 + starting_epoch, J=J, R=R, entropy=E, Q=rho, rho_prior=rho_prior, rho=rho_no_prior ,q_loss=q_loss)

        agent.save(model_path)

        if logging:
            wandb.finish()

        # logger.info('Press a button to visualize')
        # input()
        # core.evaluate(n_episodes=5, render=True)


if __name__ == '__main__':
    # print("test Prior")
    experiment(alg=SAC,
              n_epochs=0,
              starting_epoch=0,
              n_steps=0,
              n_steps_test=3000,
              tasks=["walk"],
              model=["src/checkpoint/walker/test"],
               test=True
              )

    runs = np.arange(5)
    # print("Train Prior")
    # experiment(alg=SAC,
    #            n_epochs=20,
    #            n_steps=4000,
    #            starting_epoch=0,
    #            n_steps_test=3000,
    #            tasks=["walk"],
    #            logging=False,
    #            model_name="test",
    #            model_name_logging="test",
    #            model_dir="src/checkpoint/walker",
    #            use_prior=False
    #            )
    #
    # print("Train residual")
    # model_name = "sac_residual_mass_20_kl"
    # for run in runs:
    #     experiment(alg=SAC,
    #                n_epochs=80,
    #                starting_epoch=19,
    #                n_steps=4000,
    #                n_steps_test=3000,
    #                tasks=["run"],
    #                model=["src/checkpoint/walker/test"],
    #                logging=False,
    #                model_name="{}_{}".format(model_name,run),
    #                model_name_logging="{}".format(model_name),
    #                model_dir="src/checkpoint/walker",
    #                use_prior=True,
    #                use_kl_on_pi=True,
    #                transfer=True
    #                )

    # print("Train transfer")
    # model_name = "sac_transfer_mass_20"
    # for run in runs:
    #     experiment(alg=SAC,
    #                n_epochs=80,
    #                starting_epoch=19,
    #                n_steps=4000,
    #                n_steps_test=3000,
    #                tasks=["walk"],
    #                model=["src/checkpoint/walker/sac_walk"],
    #                logging=True,
    #                model_name="{}_{}".format(model_name,run),
    #                model_name_logging="{}".format(model_name),
    #                model_dir="src/checkpoint/walker",
    #                use_prior=False,
    #                use_kl_on_pi=False,
    #                transfer=True
    #                )


    # print("Train Baseline")
    # for run in runs:
    #
    #     experiment(alg=SAC,
    #                n_epochs=100,
    #                n_steps=4000,
    #                starting_epoch=0,
    #                n_steps_test=3000,
    #                tasks=["walk"],
    #                logging=True,
    #                model_name="sac_walker_mass_20",
    #                model_name_logging="sac_walker_mass_20",
    #                model_dir="src/checkpoint/walker"
    #                )

