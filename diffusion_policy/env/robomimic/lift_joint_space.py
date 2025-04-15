from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.lift import Lift


class LiftJointSpace(Lift):
    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.
        Args:
            action (np.array): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        self.sim.data.ctrl[:] = action