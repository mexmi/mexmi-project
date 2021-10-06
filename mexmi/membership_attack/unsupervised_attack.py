
from base_sss import SubsetSelectionStrategy
from scipy.stats import entropy
import numpy as np


class UnsupervisedMemberAttack(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, tolerant_rate=0.05, unsuperY=None):
        super(UnsupervisedMemberAttack, self).__init__(size, Y_vec)
        self.tolerant_rate = tolerant_rate
        self.unsuperY = unsuperY


    def get_subset(self):
        # entropies = np.array([entropy(yv) for yv in self.Y_vec])
        print('using unsuprevised memebership attack')
        print("Y_vec", self.Y_vec[0])
        confidence = np.array([yv.max() for yv in self.Y_vec])
        sort_idx = np.argsort(confidence*(-1))
        # start_p = np.floor((len(confidence) - self.size) * self.uncertain_rate).astype(int)
        # print("self.unsuperY:", self.unsuperY)
        if self.unsuperY is not None:
            referConfidence = np.array([np.max(uy) for uy in self.unsuperY])
            referConfidence.squeeze()
            referY = np.sort(referConfidence)  # 0.1,0.3,0.5,0.9

            threshld = referY[int((1 - self.tolerant_rate) * len(referY))]
            print("threshold:", threshld)
        else:
            threshld=0.9
            print("threshold:", threshld)
        idx_chosen=[]

        for i in sort_idx:
            if confidence[i] >= threshld:
                idx_chosen.append(i)
        print("idx_chosen", len(idx_chosen))
        if len(idx_chosen) < self.size:
            print("idx_chosen is not long enough,", len(idx_chosen))
            idx_chosen = sort_idx[:self.size]
        return idx_chosen[-len(idx_chosen):]