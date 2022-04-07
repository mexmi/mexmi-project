
from base_sss import SubsetSelectionStrategy
import base_sss
import random

class RandomSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, previous_s=None):
        self.previous_s = previous_s
        super(RandomSelectionStrategy, self).__init__(size, Y_vec)
        
    
    def get_subset(self):
        # random.setstate(base_sss.sss_random_state)
        if self.previous_s is not None:
            Y_e = [self.Y_vec[ie] for ie in self.previous_s]
        else:
            Y_e = self.Y_vec
        s = random.sample([i for i in range(len(Y_e))], self.size)
        base_sss.sss_random_state = random.getstate()
        return s
