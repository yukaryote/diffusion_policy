from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport


class TransportJointSpace(TwoArmTransport):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)