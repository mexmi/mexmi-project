import config as cfg
import numpy as np
import random

random.seed(cfg.seed)
sss_random_state = random.getstate()
sss_rs = np.random.RandomState(cfg.seed)

class SubsetSelectionStrategy(object):
    def __init__(self, size, Y_vec):
        self.Y_vec      = Y_vec
        self.size       = size
