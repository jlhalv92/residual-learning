from copy import deepcopy

import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.rl_utils.replay_memory import PrioritizedReplayMemory, ReplayMemory
from mushroom_rl.rl_utils.parameters import to_parameter


class AbstractDQN(Agent):
    def __init__(self, mdp_info, policy, approximator, approximator_params, batch_size, target_update_frequency,
                 replay_memory=None, initial_replay_size=500, max_replay_size=5000, fit_params=None,
                 predict_params=None, clip_reward=False, gradient_steps=1, tau=0.01):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            approximator_params (dict): parameters of the approximator to
                build;
            batch_size ([int, Parameter]): the number of samples in a batch;
            target_update_frequency (int): the number of samples collected
                between each update of the target network;
            replay_memory ([ReplayMemory, PrioritizedReplayMemory], None): the
                object of the replay memory to use; if None, a default replay
                memory is created;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            fit_params (dict, None): parameters of the fitting algorithm of the
                approximator;
            predict_params (dict, None): parameters for the prediction with the
                approximator;
            clip_reward (bool, False): whether to clip the reward or not.

        """
        super().__init__(mdp_info, policy, backend='numpy')

        self._fit_params = dict() if fit_params is None else fit_params
        self._predict_params = dict() if predict_params is None else predict_params
        self.Q = [0.]
        self._tau = to_parameter(tau)
        self._gradient_steps = gradient_steps
        self._batch_size = to_parameter(batch_size)
        self._clip_reward = clip_reward
        self._target_update_frequency = target_update_frequency

        if replay_memory is not None:
            self._replay_memory = replay_memory["class"](mdp_info, self.info, initial_size=initial_replay_size,
                                                         max_size=max_replay_size, **replay_memory["params"])
            if isinstance(self._replay_memory, PrioritizedReplayMemory):
                self._fit = self._fit_prioritized
            else:
                self._fit = self._fit_standard
        else:
            self._replay_memory = ReplayMemory(mdp_info, self.info, initial_replay_size, max_replay_size)
            self._fit = self._fit_standard

        self._n_updates = 0

        apprx_params_train = deepcopy(approximator_params)
        apprx_params_target = deepcopy(approximator_params)

        self._initialize_regressors(approximator, apprx_params_train, apprx_params_target)

        policy.set_q(self.approximator)

        self._add_save_attr(
            _fit_params='pickle',
            _predict_params='pickle',
            _batch_size='mushroom',
            _n_approximators='primitive',
            _clip_reward='primitive',
            _target_update_frequency='primitive',
            _replay_memory='mushroom',
            _n_updates='primitive',
            approximator='mushroom',
            target_approximator='mushroom'
        )

    def fit(self, dataset):
        for _ in range(self._gradient_steps):
            self._fit(dataset)

            self._n_updates += 1
        if self._n_updates % self._target_update_frequency == 0:
            self._update_target()

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:

            state, action, reward, next_state, absorbing, _ = self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            self.Q.append(q.mean())
            self.approximator.fit(state, action, q, **self._fit_params)

    def _fit_prioritized(self, dataset):
        self._replay_memory.add(
            dataset, np.ones(len(dataset)) * self._replay_memory.max_priority)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _, idxs, is_weight = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next
            td_error = q - self.approximator.predict(state, action, **self._predict_params)

            self._replay_memory.update(td_error, idxs)

            self.approximator.fit(state, action, q, weights=is_weight,
                                  **self._fit_params)

    def _initialize_regressors(self, approximator, apprx_params_train, apprx_params_target):
        self.approximator = Regressor(approximator, **apprx_params_train)
        self.target_approximator = Regressor(approximator, **apprx_params_target)
        self._update_target()

    # def _update_target(self):
    #     """
    #     Update the target network.
    #
    #     """
    #     self.target_approximator.set_weights(self.approximator.get_weights())

    def load_transfer(self, transfer_agent):
        self.transfer_agent_aproximator = transfer_agent.target_approximator
        prior_weights = self.transfer_agent_aproximator.get_weights()
        self.approximator.set_weights(prior_weights)
        self.target_approximator.set_weights(prior_weights)
        print("[INFO]------------------------------Weights loaded")

    def _update_target(self):

        weights = self._tau() * self.approximator.get_weights()
        weights += (1 - self._tau.get_value()) * self.target_approximator.get_weights()
        self.target_approximator.set_weights(weights)

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Maximum action-value for each state in ``next_state``.

        """
        raise NotImplementedError

    def _post_load(self):
        if isinstance(self._replay_memory, PrioritizedReplayMemory):
            self._fit = self._fit_prioritized
        else:
            self._fit = self._fit_standard

        self.policy.set_q(self.approximator)

    def set_logger(self, logger, loss_filename='loss_Q'):
        """
        Setter that can be used to pass a logger to the algorithm

        Args:
            logger (Logger): the logger to be used by the algorithm;
            loss_filename (str, 'loss_Q'): optional string to specify the loss filename.

        """
        self.approximator.set_logger(logger, loss_filename)


class BoostedDQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.prev_q = None
        self.prior_Q = [0]
        self.rho = [0]
        self.boosting = False

    def setup_boosting(self, prev_qs):
        self.boosting = True
        self.prev_qs = [prev_q.target_approximator for prev_q in prev_qs]
    
    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            
            q_next = self._next_q(next_state, absorbing)

            q = reward + self.mdp_info.gamma*(q_next)

            self.Q.append(q.mean())

            prev_qs = np.zeros(q.shape)
            # rho = np.zeros(q.shape)
            if self.boosting:
                for prev_q in self.prev_qs:
                    prev_qs += prev_q.predict(state, action.astype(np.int64))

                rho = q - prev_qs

            else:
                rho = q


            self.prior_Q.append(prev_qs.mean())
            self.rho.append(rho.mean())

            self.approximator.fit(state, action, rho, **self._fit_params)

    def _next_q(self, next_state, absorbing):
        rho = self.target_approximator.predict(next_state, **self._predict_params)

        old_rhos = np.zeros(rho.shape)
        if self.boosting:
            for prev_q in self.prev_qs:
                old_rhos += prev_q.predict(next_state,**self._predict_params)

        q = rho + old_rhos
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)


class ResNetBoostedDQN(AbstractDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_q = None
        self.prior_Q = [0]
        self.rho = [0]

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)

    def setup_boosting(self, prev_q):

        self.prev_q = prev_q.target_approximator
        self.copy_weights_critic(prev_q.target_approximator[0], self.target_approximator[0])
        self.copy_weights_critic(prev_q.target_approximator[0], self.approximator[0])


    def copy_weights_critic(self, old_critic, current_critic):

        new_state_dict = current_critic.network.state_dict()
        old_state_dict = old_critic.network.state_dict()
        keys = list(old_state_dict.keys())[:-2]
        # print(keys)

        keys_freeze = list(new_state_dict.keys())[:-4]

        filtered_dict = {key: old_state_dict[key] for key in keys}
        new_state_dict.update(filtered_dict)
        current_critic.network.load_state_dict(new_state_dict, strict=False)

        for name, param in current_critic.network.named_parameters():
            if name in keys_freeze:

                param.requires_grad = False


class RestNetResidualDQN(AbstractDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_q = None
        self.prior_Q = [0]
        self.rho = [0]

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)

    def setup_boosting(self, prev_q):

        self.prev_q = prev_q.target_approximator
        self.copy_weights_critic(prev_q.target_approximator[0], self.target_approximator[0])
        self.copy_weights_critic(prev_q.target_approximator[0], self.approximator[0])


    def copy_weights_critic(self, old_critic, current_critic):

        new_state_dict = current_critic.network.state_dict()
        old_state_dict = old_critic.network.state_dict()
        keys = list(old_state_dict.keys())[:-2]
        # keys_freeze = list(new_state_dict.keys())[:-4]
        filtered_dict = {key: old_state_dict[key] for key in keys}
        new_state_dict.update(filtered_dict)
        current_critic.network.load_state_dict(new_state_dict, strict=False)

        for name, param in current_critic.network.named_parameters():
            if name in keys:

                param.requires_grad = False


class ResNetBoostedResidualDQN(AbstractDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_q = None
        self.prior_Q = [0]
        self.rho = [0]
        self.boosting = False

    def setup_boosting(self, prev_qs):

        self.prev_qs = [prev_q.target_approximator for prev_q in prev_qs]

        last_prev_qs = self.prev_qs[-1]

        self.boosting = True
        self.copy_weights_critic(last_prev_qs[0], self.target_approximator[0])
        self.copy_weights_critic(last_prev_qs[0], self.approximator[0])

    def copy_weights_critic(self, old_critic, current_critic):

        new_state_dict = current_critic.network.state_dict()
        old_state_dict = old_critic.network.state_dict()
        keys = list(old_state_dict.keys())[:-2]
        filtered_dict = {key: old_state_dict[key] for key in keys}
        new_state_dict.update(filtered_dict)
        current_critic.network.load_state_dict(new_state_dict, strict=False)

        for name, param in current_critic.network.named_parameters():
            if name in keys:
                param.requires_grad = False


    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)

            q = reward + self.mdp_info.gamma * (q_next)

            self.Q.append(q.mean())
            prev_qs = np.zeros(q.shape)

            if self.boosting:
                for prev_q in self.prev_qs:
                    prev_qs += prev_q.predict(state, action.astype(np.int64), **self._predict_params)

                q-= prev_qs

            self.prior_Q.append(prev_qs.mean())
            self.rho.append(q.mean())

            self.approximator.fit(state, action, q, **self._fit_params)

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        old_rhos = np.zeros(q.shape)

        if self.boosting:
            for prev_q in self.prev_qs:
                old_rhos += prev_q.predict(next_state, **self._predict_params)

            q += old_rhos
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)


class BoostedResidualDQN(AbstractDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_q = None
        self.prior_Q = [0]
        self.rho = [0]
        self.boosting = False

    def setup_boosting(self, prev_qs):

        self.prev_qs = [prev_q.target_approximator for prev_q in prev_qs]

        self.boosting = True

    def _fit_standard(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._next_q(next_state, absorbing)

            q = reward + self.mdp_info.gamma * (q_next)

            self.Q.append(q.mean())

            prev_qs = np.zeros(q.shape)
            # rho = np.zeros(q.shape)
            if self.boosting:
                for prev_q in self.prev_qs:
                    prev_qs += prev_q.predict(state, action.astype(np.int64))
                q-= prev_qs


            self.prior_Q.append(prev_qs.mean())
            self.rho.append(q.mean())

            self.approximator.fit(state, action, q, **self._fit_params)

    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        old_rhos = np.zeros(q.shape)

        if self.boosting:
            for prev_q in self.prev_qs:
                old_rhos += prev_q.predict(next_state, **self._predict_params)
            q += old_rhos
        if absorbing.any():
            q *= 1 - absorbing.reshape(-1, 1)

        return q.max(1)