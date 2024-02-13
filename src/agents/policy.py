
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from mushroom_rl.policy import Policy
from mushroom_rl.rl_utils.parameters import Parameter, to_parameter

class TDPolicy(Policy):
    def __init__(self, policy_state_shape=None):
        """
        Constructor.

        """
        super().__init__(policy_state_shape)

        self._approximator = None
        self._predict_params = dict()

        self._add_save_attr(_approximator='mushroom!',
                            _predict_params='pickle')

    def set_q(self, approximator):
        """
        Args:
            approximator (object): the approximator to use.

        """
        self._approximator = approximator

    def get_q(self):
        """
        Returns:
             The approximator used by the policy.

        """
        return self._approximator


class EpsGreedy(TDPolicy):
    """
    Epsilon greedy policy.

    """
    def __init__(self, epsilon, policy_state_shape=None, rhos=None, boosting=False,**kwargs):
        """
        Constructor.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates
                the probability of performing a random actions in the current
                step.

        """
        # self.explore_old_steps = explore_old_steps
        # self.explore_old = explore_old
        self.counter = 0
        # self.mode=mode
        self.boosting = boosting

        if rhos:
            self._rhos = [rho.approximator for rho in rhos]
        else:
            self._rhos = None

        super().__init__(policy_state_shape)

        self._epsilon = to_parameter(epsilon)

        self._add_save_attr(_epsilon='mushroom')

    def __call__(self, *args):
        raise "Not implemented"

    def draw_action(self, state, policy_state=None):

        if np.random.uniform() < self._epsilon(state):

            cur_action = np.array([np.random.choice(self._approximator.n_actions)])
        else:
            if self.boosting:
                q = self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params).ravel()

                for rho in self._rhos:
                    q += rho.predict(np.expand_dims(state, axis=0), **self._predict_params).ravel()

                cur_action = np.argwhere(q == np.max(q)).ravel()

            else:

                q = self._approximator.predict(np.expand_dims(state, axis=0), **self._predict_params).ravel()

                cur_action = np.argwhere(q == np.max(q)).ravel()

        if len(cur_action) > 1:
            cur_action = np.array([np.random.choice(cur_action)])

        return cur_action, None

    def set_mode(self, mode):
        self.mode = mode
    def set_epsilon(self, epsilon):
        """
        Setter.

        Args:
            epsilon ([float, Parameter]): the exploration coefficient. It indicates the
            probability of performing a random actions in the current step.

        """
        self._epsilon = to_parameter(epsilon)

    def update(self, *idx):
        """
        Update the value of the epsilon parameter at the provided index (e.g. in
        case of different values of epsilon for each visited state according to
        the number of visits).

        Args:
            *idx (list): index of the parameter to be updated.

        """
        self._epsilon.update(*idx)