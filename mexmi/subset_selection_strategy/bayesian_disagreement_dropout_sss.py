from base_sss import SubsetSelectionStrategy
import base_sss
import random
import torch

class BALDDropoutStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, n_drop=10, previous_s=None):
        self.previous_s = previous_s
        self.n_drop = n_drop
        super(BALDDropoutStrategy, self).__init__(size, Y_vec)
    
    def get_subset(self):
        # random.setstate(base_sss.sss_random_state)
        if self.previous_s is not None:
            Y_e = [self.Y_vec[ie] for ie in self.previous_s]
        else:
            Y_e = self.Y_vec #unlabelled copy mdeol outputs
        #dropout
        probs = torch.zeros([self.n_drop, len(Y_e), 10]) #np.unique(Y)
        for i in range(self.n_drop):
            for idxs in range(len(Y_e)):
                probs[i][idxs] += Y_e[idxs]
        pb = probs.mean(0)
        entropy1 = (-pb * torch.log(pb)).sum(1)
        entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)
        U = entropy2 - entropy1
        points = U.sort()[1][:self.size]
        # print("points,", points)
        if self.previous_s is not None:
            final_points = [self.previous_s[p] for p in points]
        else:
            final_points = points
        return final_points
