"""

"""
from torch.autograd import Variable
from tqdm import tqdm

from base_sss import SubsetSelectionStrategy
import base_sss
import random
import numpy as np
import torch
import torch.nn.functional as F

class AdversarialDeepFoolStrategy(SubsetSelectionStrategy):
    def __init__(self, size, X, Y_vec, copy_model, max_iter=50, previous_s=None):
        self.max_iter = max_iter
        self.X = X
        self.copy_model = copy_model.get_model() #this means the copy model
        self.previous_s = previous_s
        super(AdversarialDeepFoolStrategy, self).__init__(size, Y_vec)

    def cal_dis(self, x, y):
        x = torch.tensor(x)
        nx = torch.unsqueeze(x, 0).to('cuda')
        # print("x.shape", x.shape)
        # nx = Variable(nx, requires_grad=True)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape).to('cuda')
        out = self.copy_model((nx+eta))#torch.tensor(y)# here we assume it is numpy#= self.copy_model(nx+eta) #e1
        if isinstance(out, tuple):
            out = out[0]
        out = F.softmax(out, dim=1)
        # out = Variable(out, requires_grad=True)
        n_class = 10#out.shape[1]
        ny = py = np.argmax(out[0].cpu().detach().numpy()) #positive y and negtive y
        # print("ny,py:", ny, ",", py)
        # py = out.max(1)[1].item()
        # ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            # print("nx.grad", nx.grad)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.cpu().item()) / np.linalg.norm(wi.cpu().numpy().flatten())

                if value_i <= value_l:
                    ri = value_i/np.linalg.norm(wi.cpu().numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            query_inp = (nx+eta).to('cuda')
            out = self.copy_model(query_inp) #, e1
            if isinstance(out, tuple):
                out = out[0]
            out = F.softmax(out, dim=1)
            # print("out.shape,", out.shape)
            out.squeeze()
            py = np.argmax(out[0].cpu().detach().numpy())
            i_iter += 1

        return (eta*eta).sum()
    
    def get_subset(self):
        # random.setstate(base_sss.sss_random_state)
        if self.previous_s is not None:
            Y_e = [self.Y_vec[int(ie)] for ie in self.previous_s]#index
            X = [self.X[int(ie)] for ie in self.previous_s]
            self.X = X
        else:
            Y_e = self.Y_vec

        dis = np.zeros(len(Y_e)) #deep fool distances
        with tqdm(total = len(self.X)) as bar:
            for i in range(len(self.X)):
                # print('adv{}/{}'.format(i, len(Y_e)))
                x = self.X[i]
                y = self.Y_vec[i]
                dis[i] = self.cal_dis(x, y)  # x is the input label
                bar.update(1)

        s = dis.argsort()[:self.size]

        if self.previous_s is not None:
            s_final = [self.previous_s[int(si)] for si in s] #index
        else:
            s_final = s #Index
        return s_final #index