from mushroom_rl.utils.callbacks.callback import Callback
import numpy as np


class CurriculumWlaker(Callback):
    """
    This callback can be used to collect the maximum action value in a given
    state at each call.

    """
    def __init__(self, agent, list_tasks):
        """
        Constructor.

        Args:
            approximator ([Table, EnsembleTable]): the approximator to use;
            state (np.ndarray): the state to consider.

        """
        self.agent = agent
        self.start_curriculum = False


        super().__init__()

    def __call__(self, dataset):
        if self.start_curriculum:
            self.agent.nominal = True

    def start_curriculum(self):
        self.start_curriculum = True
