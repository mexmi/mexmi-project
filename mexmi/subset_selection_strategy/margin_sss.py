from base_sss import SubsetSelectionStrategy
import base_sss
import random
import numpy as np

class MarginSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, previous_s=None):
        self.previous_s = previous_s
        super(MarginSelectionStrategy, self).__init__(size, Y_vec)
        
    
    def get_subset(self):
        if self.previous_s is not None:
            Y_e = [self.Y_vec[int(ie)] for ie in self.previous_s]
        else:
            Y_e = self.Y_vec
        #got_Y_e
        margin_matix = np.sort(Y_e, axis=1)[:, -2:]
        margin = margin_matix[:, 1]-margin_matix[:, 0]
        s = np.argsort(margin)[: self.size]
        base_sss.sss_random_state = random.getstate()
        if self.previous_s is not None:
            final_s=[self.previous_s[e] for e in s]
        else:
            final_s=s
        return final_s