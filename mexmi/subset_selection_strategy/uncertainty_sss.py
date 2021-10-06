
from base_sss import SubsetSelectionStrategy
from scipy.stats import entropy
import numpy as np

class UncertaintySelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, previous_s=None):
        self.previous_s = previous_s
        super(UncertaintySelectionStrategy, self).__init__(size, Y_vec)
        
    def get_subset(self):
        # entropies = np.array([entropy(yv) for yv in self.Y_vec])
        # return np.argsort(entropies*-1)[:self.size]
        if self.previous_s is not None:
            Y_e = [self.Y_vec[int(ie)] for ie in self.previous_s]
        else:
            Y_e = self.Y_vec
        entropies = np.array([entropy(yv) for yv in Y_e])#self.Y_vec
        points = np.argsort(entropies*-1)[:self.size]
        if self.previous_s is not None:
            final_points = [self.previous_s[p] for p in points]
        else:
            final_points = points
        return final_points