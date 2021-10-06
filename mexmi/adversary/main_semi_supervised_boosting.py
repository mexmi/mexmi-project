#!/usr/bin/python
"""This file is the step to perform semi-supervised boosting module after the attacker
   finished iteration training.

"""

import argparse
import copy
import json
import os
import os.path as osp
import pickle
# from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import classifier
import mexmi.config as cfg
import mexmi.utils.model_scheduler as model_utils
import mexmi.utils.model_scheduler_semi as model_utils_semi
# import mexmi.utils.split_model as split_model_utils
# from bayes_attack import BayesMemberAttack
import norm
import parser_params
import splitnet
from adversarial_deepfool_sss import AdversarialDeepFoolStrategy
from autoaugment import CIFAR10Policy
from bayesian_disagreement_dropout_sss import BALDDropoutStrategy
from generative_attack import GenerativeMemberAttack
from gradient_attack import GradientMemberAttack
from graph_density_sss import GraphDensitySelectionStrategy
from kcenter_sss import KCenterGreedyApproach
from mexmi import datasets
import mexmi.models.zoo as zoo
from mexmi.victim.blackbox import Blackbox
from tqdm import tqdm
import random

from margin_sss import MarginSelectionStrategy
from random_sss import RandomSelectionStrategy
from shadow_model_attack import ShadowModelMemberAttack, ConfShadowModelMemberAttack, AdjShadowModelMemberAttack
from uncertainty_sss import UncertaintySelectionStrategy
from unsupervised_attack import UnsupervisedMemberAttack
from utils.utils import clipDataTopX
import time
import datetime

class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def samples_to_transferset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))

def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer

class GetAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=8):
        self.blackbox = blackbox
        self.attack_model = None
        self.attack_device = None
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()
        self.q_idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]
        self.transfery = []
        self.pre_idxset = []
        self._restart()
        self.sampling_method = 'random'
        self.ma_method = 'unsupervised'

        self.no_training_in_initial = 0
        self.shadow_idx = None
        self.unsuperY = None


    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset))) #idx根据queryset进行更改的
        self.transferset = []
        self.transfery = []

    def set_attack_model(self, attack_model, device):
        # self.attack_model = Blackbox(attack_model, device, 'probs')
        self.attack_model = attack_model
        self.attack_device = device
        # self.attack_model.eval()

    # For KCenter
    def get_initial_centers(self):
        Y_vec_true = []
        print("get_initial_centers")
        assert self.attack_model is not None, "attack_model made a mistake!"
        for b in range(int(np.ceil(len(self.pre_idxset)/self.batch_size))): #不是这么回事
            # print("b = ", b)
            # print("pre_dixset = ", self.pre_idxset)
            x_idx = self.pre_idxset[(b * self.batch_size): min(((b+1) * self.batch_size), len(self.pre_idxset))]
            # print("x_idx:", x_idx)
            trX = torch.stack([self.queryset[int(i)][0] for i in x_idx]).to(self.attack_device)
            trY = self.attack_model(trX).cpu()
            Y_vec_true.append(trY)
        Y_vec_true = np.concatenate(Y_vec_true)
        # print("in get_initial_centers:,len(x_idx):", len(Y_vec_true))
        # print("Y_vec_true,", Y_vec_true.shape)
        return Y_vec_true

    def set_unsuperY(self):
        unsuperY = []
        unsuper_idx_set = set(range(cfg.unsuper_data_start, cfg.unsuper_data_start + int(cfg.unsuper_data)))  # 184800
        for u in range(int(np.ceil(len(unsuper_idx_set) / self.batch_size))):
            # download_x
            # idx来自idx_set
            uidx = list(unsuper_idx_set)[
                   u * self.batch_size: min((1 + u) * self.batch_size, len(unsuper_idx_set))]
            # print("uidx:", uidx)
            x_u = torch.stack([self.queryset[iu][0] for iu in uidx]).to(self.attack_device)
            y_u = self.attack_model(x_u).cpu()  # not training data.
            # X_rest.append(x_u.cpu())
            # yu_top_1 = clipDataTopX(y_u, top=1)
            # unsuperY.append(yu_top_1)
            unsuperY.append(y_u.numpy())
            # just get unsupery
        self.unsuperY = np.concatenate(unsuperY)

    def get_transferset(self, k, sampling_method='random', ma_method='unsupervised', pre_idxset=[],
                        shadow_attack_model=None, device=None, second_sss=None, it=None, initial_seed=[]):
        self.sampling_method=sampling_method
        self.ma_method=ma_method
        print("pre_idxset:", len(pre_idxset))
        start_B = 0
        end_B = k
        dt=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt)
        with tqdm(total=k) as pbar:
            if self.sampling_method == 'initial_seed' or self.sampling_method == 'training_data' or\
                self.sampling_method == 'label':
                for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                    # print("start", B)
                    if self.sampling_method == 'training_data':
                        self.q_idx_set = set(range(60000))
                        self.q_idx_set.intersection_update(self.idx_set)
                    else:
                        self.q_idx_set = copy.copy(self.idx_set)

                    idxs = np.random.choice(list(self.q_idx_set), replace=False,
                                            size=(self.batch_size, len(self.q_idx_set)))  # 8，200-目前拥有的transferset的大小。
                    print("initial_seed_idxs", idxs)
                    #这就是选出的idx了
                    for index in idxs:
                        if index >= cfg.trainingbound:
                            self.no_training_in_initial += 1

                    self.idx_set = self.idx_set - set(idxs)
                    #这里存储 选出的pre_idxset
                    self.pre_idxset = np.append(self.pre_idxset, idxs)
                    # print("idx,", idxs)
                    if len(self.idx_set) == 0:
                        print('=> Query set exhausted. Now repeating input examples.')
                        self.idx_set = set(range(len(self.queryset)))

                    x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                    y_t = self.blackbox(x_t).cpu()

                    #目前全部按照ChainDataset来说
                    if hasattr(self.queryset, 'samples'):
                        # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                        img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                    else:
                        # Otherwise, store the image itself # But, we need to store the non-transformed version
                        # img_t = [self.queryset.data[i] for i in idxs]
                        if len(self.queryset.data) <= 3:
                            img_t, gt_label = self.queryset.getitemsinchain(idxs)
                        else:
                            img_t = [self.queryset.data[i] for i in idxs]
                            # gt_label = [self.queryset.targets[i] for i in idxs]
                        # if isinstance(self.queryset.data[0], torch.Tensor):
                        #     img_t = [x.numpy() for x in img_t]

                    for i in range(len(idxs)):
                        img_t_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                        img_t_i = img_t_i.squeeze() if isinstance(img_t_i, np.ndarray) else img_t_i
                        self.transferset.append((img_t_i, y_t[i].squeeze()))
                        self.transfery.append(y_t[i].squeeze().numpy())
                    pbar.update((x_t.size(0)))
            elif self.sampling_method == 'use_default_initial':
                assert len(initial_seed) > 0, 'has no input initial seed!'
                chosed_idx = [list(self.idx_set)[int(e)] for e in initial_seed]  # 让idx_set减去这个chosed_idx；已经做出了选择
                self.idx_set = self.idx_set - set(chosed_idx)
                self.pre_idxset = np.append(self.pre_idxset, chosed_idx)
                print("self.pre_idxset:", self.pre_idxset)
                for index in chosed_idx:
                    if index >= cfg.trainingbound:
                        self.no_training_in_initial += 1
                # Query
                for b in range(int(np.ceil(len(initial_seed) / self.batch_size))):
                    # x_b = x_t[b * self.batch_size: min((1 + b) * self.batch_size, len(s))].to(self.blackbox.device)
                    c_idx = chosed_idx[b * self.batch_size: min((1 + b) * self.batch_size, len(initial_seed))]
                    x_b = torch.stack([self.queryset[i][0] for i in c_idx]).to(self.blackbox.device)
                    y_b = self.blackbox(x_b).cpu()

                    if hasattr(self.queryset, 'samples'):
                        # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                        img_t = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
                    else:
                        # Otherwise, store the image itself # But, we need to store the non-transformed version
                        if len(self.queryset.data) <= 3:
                            # print("\nwe use a mnist chain")
                            img_p, _ = self.queryset.getitemsinchain(c_idx)
                        else:
                            img_p = [self.queryset.data[i] for i in c_idx]

                    for m in range(len(c_idx)):
                        img_p_i = img_p[m].numpy() if isinstance(img_p[m], torch.Tensor) else img_p[m]
                        img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                        self.transferset.append((img_t_i, y_b[m].squeeze()))  # self.transferset
                        self.transfery.append(y_b[m].squeeze().numpy())
                    pbar.update(x_b.size(0))
            else: #这是
                print("\nuse key senter, adversarial, uncertrainty, adversarial k-center, membership_attack to choose and query data")

                #self.idx_set: means remaining data; #must set attack_model first
                assert self.attack_model is not None, "self.attack_model is None!"
                X_rest = []
                Y_copy = []
                #将每个剩余的数都经过copy model, {X,Y}
                with tqdm(total=int(np.ceil(len(self.idx_set) / self.batch_size))) as copy_bar:
                    for b in range(int(np.ceil(len(self.idx_set) / self.batch_size))):
                        # download_x
                        #idx来自idx_set
                        idx = list(self.idx_set)[b * self.batch_size: min((1 + b) * self.batch_size, len(self.idx_set))]
                        # one batch idx
                        for jj in idx:
                            assert jj not in pre_idxset, "FAILURE!"
                        x_p = torch.stack([self.queryset[i][0] for i in idx]).to(self.attack_device)
                        y_p = self.attack_model(x_p).cpu()
                        # y_p = self.blackbox(x_p).cpu()
                        X_rest.append(x_p.cpu())
                        Y_copy.append(y_p)
                        copy_bar.update(b)
                X_rest = np.concatenate(X_rest) #对应的X_idx 就是idx_set
                Y_copy = np.concatenate(Y_copy)
                #sss = SubsetSelectionStrategy
                if sampling_method == 'kcenter': #4
                    print("\nkcenter: KCenterGreedyApproach")
                    # # print("Y", Y.shape)
                    # if second_sss == 'shadow_model':
                    #     print("\n has second_sss")
                    #     # sss1 = UncertaintySelectionStrategy(k*10, Y_copy)
                    #     sss1 = KCenterGreedyApproach(k*10, Y_copy, self.get_initial_centers())  # Y是没用过的数求得的。这是一个类；
                    #     s1 = sss1.get_subset()
                    #     sss = ShadowModelMemberAttack(k, shadow_attack_model, Y_copy, previous_s=s1)
                    # else:
                    sss = KCenterGreedyApproach(k, Y_copy, self.get_initial_centers())  # Y是没用过的数求得的。这是一个类；
                elif sampling_method == 'graph_density': #3
                    sss = GraphDensitySelectionStrategy(k, Y_copy, self.get_initial_centers())
                elif sampling_method == 'margin': #2
                    print("margin, MarginSelectionStrategy")
                    sss = MarginSelectionStrategy(k, Y_copy)
                    # print("sss-kcenter:", sss)
                elif sampling_method == 'random':
                    print("random, RandomSelectionStrategy")
                    sss = RandomSelectionStrategy(k, Y_copy)
                elif sampling_method == 'adversarial_deepfool': #5
                    print("\nAdversarialDeepFoolStrategy")
                    # print("adversarial:AdversarialSelectionStrategy")
                    # sss = AdversarialSelectionStrategy(k, choosed_X, Y, X, self.attack_model) #没有完成
                    sss = AdversarialDeepFoolStrategy(size=k, X=X_rest, Y_vec=Y_copy, copy_model=self.attack_model)
                    # sss = AdversarialSelectionStrategy(k, Y_copy, X_rest, self.attack_model, device)  # 没有完成
                elif sampling_method == 'bayes_ald_dropout': #6
                    sss = BALDDropoutStrategy(k, Y_copy)
                elif sampling_method == 'uncertainty': #1
                    print("\nuncertainty:UncertaintySelectionStrategy")
                    if second_sss == 'shadow_model':
                        print("\n has second_sss")
                        sss1 = UncertaintySelectionStrategy(10000, Y_copy)#k*10
                        s1 = sss1.get_subset()
                        sss = ShadowModelMemberAttack(k, shadow_attack_model, Y_copy, previous_s=s1)
                    else:
                        sss = UncertaintySelectionStrategy(k, Y_copy)

                elif self.sampling_method == 'membership_attack':
                    if self.ma_method == 'unsupervised':
                        print("\nuse unsupervised - mia to choose and query data")
                        # ma_unsupervised_sampling_method()
                        # assert unsuperset is not None, 'Ussuperset is none!'
                        # unsuperY = []
                        # unsuper_idx_set = set(range(35000, 35000 + int(cfg.unsuper_data)))  # 184800
                        # for u in range(int(np.ceil(len(unsuper_idx_set) / self.batch_size))):
                        #     # download_x
                        #     # idx来自idx_set
                        #     uidx = list(unsuper_idx_set)[
                        #            u * self.batch_size: min((1 + u) * self.batch_size, len(unsuper_idx_set))]
                        #     # print("uidx:", uidx)
                        #     x_u = torch.stack([self.queryset[iu][0] for iu in uidx]).to(self.attack_device)
                        #     y_u = self.attack_model(x_u).cpu()  # not training data.
                        #     # X_rest.append(x_u.cpu())
                        #     # yu_top_1 = clipDataTopX(y_u, top=1)
                        #     # unsuperY.append(yu_top_1)
                        #     unsuperY.append(y_u.numpy())
                        #     # just get unsupery
                        # self.unsuperY = np.concatenate(unsuperY)
                        sss1 = UnsupervisedMemberAttack(size=5 * k, Y_vec=Y_copy, tolerant_rate=0.15,
                                                        unsuperY=self.unsuperY)  # Y from copy
                        s1 = sss1.get_subset()

                    elif self.ma_method == 'shadow_model':
                        # print("\nuse shadow_model mia to choose and query data")
                        # sss = ShadowModelMemberAttack(size=k, shadow_attack_model=shadow_attack_model,
                        #                               Y_vec=Y_copy, batch_size=10)

                        print("\nuse shadow_model mia to choose and query data")
                        sss1 = AdjShadowModelMemberAttack(size=k*15, shadow_attack_model=shadow_attack_model,
                                                       Y_vec=Y_copy, batch_size=10, transfery=self.transfery, it=it)
                        s1, choosen_idx_10, conf_idx_10 = sss1.get_subset()
                    else:
                        print("\nwrong sampling_method/membership inference setting")

                    s2 = np.asarray(s1)
                    s2 = np.reshape(s2, [1, -1])
                    s2 = s2.squeeze()
                    self.shadow_idx = [list(self.idx_set)[int(e)] for e in s2]
                    print("self.shadow_idx", len(self.shadow_idx))
                    print("shadow_idx,", np.sum(np.asarray(self.shadow_idx) < cfg.trainingbound))

        print('self.idx_set', len(self.idx_set))
        print('self.pre_idxset', len(self.pre_idxset))
        dt1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("start_query_strategy:", dt)
        print("start_query_strategy:", dt1)
        return self.transferset, self.pre_idxset

    def get_unlabled_transferset(self):
        # shadow_idx
        # here add the vmi positive set, thus must use vmi_function
        self.unlabeled_transferset=[]
        print("----shadow_idx", len(self.shadow_idx))

        for b in range(int(np.ceil(len(self.shadow_idx) / self.batch_size))):
            c_idx = self.shadow_idx[b * self.batch_size: min((1 + b) * self.batch_size, len(self.shadow_idx))]

            if hasattr(self.queryset, 'samples'):
                # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                img_p = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
            else:
                # Otherwise, store the image itself # But, we need to store the non-transformed version
                # img_t = [self.queryset.data[i] for i in c_idx]
                # img_t = self.queryset.getitemsinchain(c_idx)
                if len(self.queryset.data) <= 3 and len(self.queryset.data) > 1:
                    img_p, _ = self.queryset.getitemsinchain(c_idx)
                else:
                    img_p = [self.queryset.data[i] for i in c_idx]

            for m in range(len(c_idx)):
                img_p_i = img_p[m].numpy() if isinstance(img_p[m], torch.Tensor) else img_p[m]
                img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                y = torch.zeros([10])
                self.unlabeled_transferset.append((img_t_i, y))  # self.transferset
        return self.unlabeled_transferset

    def mia_on_query_result(self, mi_attacker=None, iter_num=0):
        # use self.pre_idxset
        # use self.queryset
        self.transferset = []
        k = 0 # k is vain
        for b in range(int(np.ceil(len(self.pre_idxset) / self.batch_size))):  # s
            # x_b = x_t[b * self.batch_size: min((1 + b) * self.batch_size, len(s))].to(self.blackbox.device)
            c_idx = self.pre_idxset[
                    b * self.batch_size: min((1 + b) * self.batch_size, len(self.pre_idxset))]  # chosed_idx
            #all choosed idx set
            x_b = torch.stack([self.queryset[int(i)][0] for i in c_idx]).to(self.blackbox.device)
            y_b = self.blackbox(x_b).cpu()
            # print("y_b,", y_b)
            if cfg.imp_vic_mem:
                if cfg.vic_mem_method == 'unsupervised':
                    print("use imp vic mem -- unsupervised")
                    # unsuperY = []
                    # unsuper_idx_set = set(range(35000, 35000 + int(cfg.unsuper_data)))  # 184800
                    # for u in range(int(np.ceil(len(unsuper_idx_set) / self.batch_size))):
                    #     # download_x
                    #     # idx来自idx_set
                    #     uidx = list(unsuper_idx_set)[
                    #            u * self.batch_size: min((1 + u) * self.batch_size, len(unsuper_idx_set))]
                    #     # print("uidx:", uidx)
                    #     x_u = torch.stack([self.queryset[iu][0] for iu in uidx]).to(self.attack_device)
                    #     y_u = self.attack_model(x_u).cpu()  # not training data.
                    #     # X_rest.append(x_u.cpu())
                    #     # yu_top_1 = clipDataTopX(y_u, top=1)
                    #     # unsuperY.append(yu_top_1)
                    #     unsuperY.append(y_u.numpy())
                    #     # just get unsupery
                    # self.unsuperY = np.concatenate(unsuperY)
                    # there is unsuper score, we set it earler
                    vsss = UnsupervisedMemberAttack(k, y_b, tolerant_rate=0.06, unsuperY=self.unsuperY)
                    # here just use a unsupervsed way- it need a pre-trained attack model
                else:
                    shadow_attack_model = mi_attacker
                    vsss = ShadowModelMemberAttack(k, shadow_attack_model, y_b)  # who
                vs = vsss.get_subset()
                # member_idx = [c_idx[int(ivs)] for ivs in vs]#vs is index
                print("the part of 'true' membership is:", len(vs) / len(c_idx))
                y_member = torch.ones(y_b.shape)
                y_member[vs] = 5
                y_bf = torch.stack((y_b, y_member), dim=1)

            if hasattr(self.queryset, 'samples'):
                # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                img_p = [self.queryset.samples[i][0] for i in c_idx]  # Image paths
            else:
                # Otherwise, store the image itself # But, we need to store the non-transformed version
                if len(self.queryset.data) <= 3 and len(self.queryset.data) > 1:
                    # print("\nwe use a mnist chain")
                    img_p, _ = self.queryset.getitemsinchain(c_idx)
                else:
                    img_p = [self.queryset.data[i] for i in c_idx]

            for m in range(len(c_idx)):
                img_p_i = img_p[m].numpy() if isinstance(img_p[m], torch.Tensor) else img_p[m]
                img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                self.transferset.append((img_t_i, y_bf[m].squeeze()))  # self.transferset

        return self.transferset

    def augment_shadow_set(self, transferset_shadow):
        augment_transferset = []
        transform_method = transforms.Compose([
            CIFAR10Policy(),
        ])
        for data, labels in transferset_shadow:
            img = Image.fromarray(data)
            new_img = np.asarray(transform_method(img))
            augment_transferset.append((data, labels))
            augment_transferset.append((new_img, labels))
        return augment_transferset

    def transfer_to_shadow_set(self, transferset_o):
        shadow_transferset=[]
        for img,probs in transferset_o:
            label = np.argmax(probs)
            y = torch.zeros([len(probs)])
            y[label] = 1.
            shadow_transferset.append((img, y))
        return shadow_transferset

    def get_transferset_shadow_out(self, shadow_out_len):
        start_B = 0
        end_B = int(shadow_out_len)
        transferset_shadow_out = []
        with tqdm(total=shadow_out_len) as pbar:
            shadow_idx_set_out = set(np.random.choice(list(self.idx_set), replace=False,
                                    size=shadow_out_len))  # 8，200-目前拥有的transferset的大小。
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(shadow_idx_set_out)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself # But, we need to store the non-transformed version
                    # img_t = [self.queryset.data[i] for i in idxs]
                    if len(self.queryset.data) <= 3:
                        img_t, gt_label = self.queryset.getitemsinchain(idxs)
                    else:
                        img_t = [self.queryset.data[i] for i in idxs]
                        # gt_label = [self.queryset.targets[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(len(idxs)):
                    img_p_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                    img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                    gt = torch.zeros([10])
                    gt[0] = 1. #这里因为有多个种类
                    transferset_shadow_out.append((img_t_i, gt))
                pbar.update(len(idxs))
        return transferset_shadow_out

    def get_transferset_shadow(self, k):
        # self.sampling_method=sampling_method
        # self.ma_method=ma_method
        # print("pre_idxset:", len(pre_idxset))
        start_B = 0
        end_B = int(k/2)
        transferset_shadow = []
        transferset_shadow2 =[]
        with tqdm(total=k) as pbar:
            shadow_idx_set = set(range(cfg.start_shadow_test, cfg.start_shadow_test+int(cfg.shadow_data/2)))#120000
            shadow_idx_set_out = set(range(cfg.start_shadow_test_out, cfg.start_shadow_test_out+int(cfg.shadow_data/2))) #124800 emnist 58750
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(shadow_idx_set)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself # But, we need to store the non-transformed version
                    # img_t = [self.queryset.data[i] for i in idxs]
                    # print("idxs in shadow data", idxs)
                    if len(self.queryset.data) <= 3:
                        img_t, gt_label = self.queryset.getitemsinchain(idxs)
                    else:
                        img_t = [self.queryset.data[i] for i in idxs]
                        gt_label = [self.queryset.targets[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(len(idxs)):
                    img_p_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                    img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                    # img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    gt = torch.zeros([10])
                    gt[gt_label[i]] = 1.
                    transferset_shadow.append((img_t_i, gt))
                pbar.update(len(idxs))
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(shadow_idx_set_out)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself # But, we need to store the non-transformed version
                    # img_t = [self.queryset.data[i] for i in idxs]
                    if len(self.queryset.data) <= 3:
                        img_t, gt_label = self.queryset.getitemsinchain(idxs)
                    else:
                        img_t = [self.queryset.data[i] for i in idxs]
                        # gt_label = [self.queryset.targets[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(len(idxs)):
                    img_p_i = img_t[i].numpy() if isinstance(img_t[i], torch.Tensor) else img_t[i]
                    img_t_i = img_p_i.squeeze() if isinstance(img_p_i, np.ndarray) else img_p_i
                    # img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    gt = torch.zeros([10])
                    gt[0] = 1. #这里因为有多个种类
                    transferset_shadow2.append((img_t_i, gt))
                pbar.update(len(idxs))
        return transferset_shadow, transferset_shadow2
    #k is the overall test samples
    def get_transferset_shadow_test(self, k):
        # self.sampling_method=sampling_method
        # self.ma_method=ma_method
        # print("pre_idxset:", len(pre_idxset))
        start_B = 0
        end_B = int(k/2)
        # X_copy = []
        # Y_copy = []
        # transferset_shadow_test = []
        # transferset_shadow2_test =[]
        y_shadow = []
        target_shadow = []
        with tqdm(total=k) as pbar:
            shadow_idx_set = set(range(cfg.start_shadow_test, cfg.start_shadow_test+int(k/2)))#120000
            shadow_idx_set_out = set(range(cfg.start_shadow_test_out, cfg.start_shadow_test_out+int(k/2))) #124800 emnist 58750
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(shadow_idx_set)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                x_s = torch.stack([self.queryset[i][0] for i in idxs]).to(self.attack_device)
                y_s = self.blackbox(x_s).cpu().numpy()

                for i in range(len(idxs)):
                    y_shadow.append(y_s[i])
                    target_shadow.append(1)
                pbar.update(len(idxs))
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(shadow_idx_set_out)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                x_s = torch.stack([self.queryset[i][0] for i in idxs]).to(self.attack_device)
                y_s = self.blackbox(x_s).cpu().numpy()
                for i in range(len(idxs)):
                    y_shadow.append(y_s[i])
                    target_shadow.append(0)

                pbar.update(len(idxs))
        return y_shadow, target_shadow

    def get_transferset_unsuper(self, k):
        start_B = 0
        end_B = int(k/2)
        transferset_unsuper = []
        with tqdm(total=k) as pbar:
            unsuper_idx_set = set(range(50000, 50000+int(cfg.unsuper_data)))#184800
            for t, B in enumerate(range(start_B, end_B, self.batch_size)):  # 1,200;步长：8
                idxs = list(unsuper_idx_set)[t*self.batch_size : min((t+1)*self.batch_size, end_B)]
                if hasattr(self.queryset, 'samples'):
                    # Any DatasetFolder (or subclass) has this attribute # Saving image paths are space-efficient
                    img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                else:
                    # Otherwise, store the image itself # But, we need to store the non-transformed version
                    # img_t = [self.queryset.data[i] for i in idxs]
                    # print("idxs in shadow data", idxs)
                    if len(self.queryset.data) <= 3:
                        img_t, gt_label = self.queryset.getitemsinchain(idxs)
                    else:
                        img_t = [self.queryset.data[i] for i in idxs]
                        gt_label = [self.queryset.targets[i] for i in idxs]
                    if isinstance(self.queryset.data[0], torch.Tensor):
                        img_t = [x.numpy() for x in img_t]

                for i in range(len(idxs)):
                    img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                    gt = torch.zeros([10])
                    # gt[gt_label[i]] = 1.
                    transferset_unsuper.append((img_t_i, gt))
                pbar.update(len(idxs))
        return transferset_unsuper

def Parser():
    parser = argparse.ArgumentParser(description='Train a model')
    # -----------------------Query arguments
    parser.add_argument('--victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"',
                        default=cfg.VICTIM_DIR)
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set',
                        default=cfg.transfer_set_out_dir)  # required=True,
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))',
                        default=cfg.queryset)  # required=True
    parser.add_argument('--root', metavar='DIR', type=str, help='Root directory for ImageFolder', default=None)
    parser.add_argument('--modelfamily', metavar='TYPE', type=str, help='Model family', default=None)

    parser.add_argument("-second_sss", default=None) #random, uncertainty, adversarial, kcenter
    # ----- assert in iterative
    parser.add_argument("-iterative", default=True, help="use iterative training method or not")
    parser.add_argument("-initial_seed", default=cfg.initial_seed, help="intial seed")  #### None; 200
    parser.add_argument("-num_iter", default=cfg.num_iter, help="num of iterations")  #### None;
    parser.add_argument("-k", default=cfg.k,
                        help="add queries")  ## 10*100 ## None; k samples are chosen in accordance with the sampling_method

    # ----------------------Required arguments
    parser.add_argument('--attack_model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle',
                        default=cfg.attack_model_dir)
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name', default=cfg.attack_model_arch)
    parser.add_argument('--testdataset', metavar='DS_NAME', type=str, help='Name of test', default=cfg.test_dataset)
    # ----------------------Optional arguments; train dataset
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=cfg.batch_size)
    parser.add_argument('-e', '--epochs', type=int, default=cfg.epoch, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=cfg.log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm',
                        choices=('sgd', 'sgdm', 'adam', 'adagrad'))

    return parser

def generateShadowAttackData(train_loader, test_loader, shadow_model, device, len_data):
    biP = []
    biY = []
    with tqdm(total=len_data) as gbar:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)  # , targets.to(device)
                outputs = shadow_model(inputs)
                p_now = outputs.cpu().squeeze().numpy()
                biP.append(p_now)
                biY.append(np.ones(len(p_now)).astype(int).tolist())
                gbar.update(len(p_now))
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)  # , targets.to(device)
                outputs = shadow_model(inputs)
                p_now = outputs.cpu().squeeze().numpy()
                biP.append(p_now)
                biY.append(np.zeros(len(p_now)).astype(int).tolist())
                gbar.update(len(p_now))

    return np.asarray(biP), np.asarray(biY)

def train_bi_classifier(shadow_model, shadow_set, shadow_set_out, batch_size=10, num_workers=10, device = 'cuda'):
    # Data loaders
    train_loader = DataLoader(shadow_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if shadow_set_out is not None:
        test_loader = DataLoader(shadow_set_out, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    shadow_model.eval()
    #Y=0/1
    #np.array, np.array
    biP, biY = generateShadowAttackData(train_loader, test_loader, shadow_model, device, len_data=2*len(shadow_set))
    biP = np.concatenate(biP)
    biP = np.squeeze(biP)
    biY = np.concatenate(biY)
    biY = np.squeeze(biY)

    print("bip", biP.shape) #(5000,10)
    print("biy", biY.shape) #(5000,)
    biP = clipDataTopX(biP, top=3)
    shadow_attack_model = classifier.train_attack_model(X_train=biP,
                                                        y_train=biY,
                                                        epochs=50,
                                                        batch_size=10,
                                                        learning_rate=0.01,
                                                        n_hidden=64,
                                                        l2_ratio=1e-6,
                                                        model='softmax') #仅仅是一个softmax模型
    return shadow_attack_model

def load_bi_classifier(shadow_model, shadow_set, shadow_set_out, batch_size=10, num_workers=10, device = 'cuda', shadow_model_path='data'):
    # Data loaders
    train_loader = DataLoader(shadow_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if shadow_set_out is not None:
        test_loader = DataLoader(shadow_set_out, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    shadow_model.eval()
    #Y=0/1
    #np.array, np.array
    biP, biY = generateShadowAttackData(train_loader, test_loader, shadow_model, device, len_data=2*len(shadow_set))
    biP = np.concatenate(biP)
    biP = np.squeeze(biP)
    biY = np.concatenate(biY)
    biY = np.squeeze(biY)

    print("bip", biP.shape) #(5000,10)
    print("biy", biY.shape) #(5000,)
    biP = clipDataTopX(biP, top=3)
    shadow_attack_model = classifier.load_attack_model(X_train=biP,
                                                        y_train=biY,
                                                        epochs=50,
                                                        batch_size=10,
                                                        learning_rate=0.01,
                                                        n_hidden=64,
                                                        l2_ratio=1e-6,
                                                        model='softmax',
                                                        shadow_model_path=shadow_model_path) #仅仅是一个softmax模型
    return shadow_attack_model

def main():
    parser = Parser()
    args = parser.parse_args()
    params = vars(args)
    # ----------- Seed, device, attack_dir
    torch.manual_seed(cfg.DEFAULT_SEED)
    np.random.seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    attack_model_dir = params['attack_model_dir']
    shadow_model_dir = cfg.shadow_model_dir
    valid_datasets = datasets.__dict__.keys()

    queryset_name = params['queryset']
    queryset_names = queryset_name.split(',')
    # print("valid_datasets: ", valid_datasets)
    for i, qname in enumerate(queryset_names):
        if qname.find("-") > -1:
            qname = qname.split("-")[0]
        if qname not in valid_datasets:  # 几大data family
            raise ValueError('Dataset not found. Valid arguments = {}, qname= {}'.format(valid_datasets, qname))
        modelfamily = datasets.dataset_to_modelfamily[qname] if params['modelfamily'] is None else params[
            'modelfamily']
        print("modelfamily,", modelfamily)
        break
    # 目前全来自一个家族MNIST
    transform_query = datasets.modelfamily_to_transforms[modelfamily]['train']
    # ----------- Set up testset
    test_dataset_name = params['testdataset']  # 用的是MNIST test
    # test_valid_datasets = datasets.__dict__.keys()
    test_modelfamily = datasets.dataset_to_modelfamily[test_dataset_name]
    test_transform = datasets.modelfamily_to_transforms[test_modelfamily]['test']
    # print("test_transform:", test_transform.__dict__.keys())

    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=test_transform)
    elif len(queryset_names) > 1:  # 拥有多个dataset
        qns = "ChainMNIST"
        for qn in queryset_names:
            if qn.find("CIFAR10") == 0:
                qns = "ChainCIFAR"
                break
        queryset = datasets.__dict__[qns](chain=queryset_names, train=True, transform=test_transform)
        # print("transform_query:", transform_query)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=test_transform)
    print("query_set:", len(queryset))

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    blackbox, num_classes = Blackbox.from_modeldir_split(blackbox_dir, device)  # 这里可以获得victim_model
    blackbox.eval()

    # ----------- Initialize adversary
    batch_size = params['batch_size']

    if test_dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    test_dataset = datasets.__dict__[test_dataset_name]
    testset = test_dataset(train=False, transform=test_transform)  # 这里是可以下载的

    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    sm_set = ['membership_attack,unsupervised,adversarial_deepfool,vmi'] #['uncertainty,,vmi']
    for sm_m in sm_set:

        exp = sm_m.split(',')
        params['sampling_method'] = exp[0]
        params['ma_method'] = exp[1]
        params['second_sss'] = exp[2]

        print("\n\n\n ----------------start {}".format(sm_m))
        # ----------- Set up queryset
        print("\ndownload queryset dataset")

        shadow_attack_model = None
        # ----------- 构建转移set
        print('=> constructing transfer set...')
        adversary = GetAdversary(blackbox, queryset, batch_size=batch_size)  # 新建了一个类
        # ----------- get #'initial_seed'个

        initial_seed = osp.join(cfg.transfer_set_out_dir, cfg.queryset,
                                "{}.npy".format(
                                    sm_m))  # "{}.npy".format(sm_m) #"initial_idxset_{}.npy".format(cfg.DEFAULT_SEED)
        pre_idxset_ = np.load(initial_seed)[:cfg.initial_seed]
        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['initial_seed'],
                                                               sampling_method='use_default_initial',
                                                               shadow_attack_model=shadow_attack_model,
                                                               initial_seed=pre_idxset_)

        # change_to_trainable_set
        transferset = samples_to_transferset(transferset_o, budget=len(transferset_o), transform=transform_query)
        print('=> Training at budget = {}'.format(len(transferset)))

        # ----------- Set up attack model
        attack_model_name = params['model_arch']
        pretrained = params['pretrained']

        # ----------- shadow_model
        criterion_train = model_utils.soft_cross_entropy
        if params['ma_method'] == 'shadow_model':

            transferset_shadow = adversary.transfer_to_shadow_set(transferset_o)[:2000]
            if cfg.augment_mi:
                print("====shadow model use augment mi")
                transferset_shadow = adversary.augment_shadow_set(transferset_shadow)

            shadow_set = samples_to_transferset(transferset_shadow, budget=len(transferset_shadow),
                                                transform=transform_query) #have transformation when training!

            shadow_set_2 = samples_to_transferset(transferset_shadow, budget=len(transferset_shadow),
                                                  transform=test_transform) #put it in again

            # shadow_queryset
            s_queryset_name = cfg.shadow_queryset
            s_queryset_names = s_queryset_name.split(',')
            print("valid_datasets: ", valid_datasets)
            for i, qname in enumerate(s_queryset_names):
                if qname.find("-") > -1:
                    qname = qname.split("-")[0]
                if qname not in valid_datasets:  # 几大data family
                    raise ValueError('Dataset not found. Valid arguments = {}, qname= {}'.format(valid_datasets, qname))

            if s_queryset_name == 'ImageFolder':
                assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
                s_queryset = datasets.__dict__[queryset_name](root=params['root'], transform=test_transform)
            elif len(s_queryset_names) > 1:  # 拥有多个dataset
                qns = "ChainMNIST"
                for qn in s_queryset_names:
                    if qn.find("CIFAR10") == 0:
                        qns = "ChainCIFAR"
                        break
                s_queryset = datasets.__dict__[qns](chain=s_queryset_names, train=True,
                                                    transform=test_transform)
                # print("transform_query:", transform_query)
            else:
                s_queryset = datasets.__dict__[queryset_name](train=True, transform=test_transform)
            print("query_set:", len(queryset))
            # transform_query = datasets.modelfamily_to_transforms[modelfamily]['train']
            s_adversary = GetAdversary(blackbox, s_queryset, batch_size=batch_size) #新建了一个类

            shadow_mode_path = osp.join(shadow_model_dir, sm_m)
            if not osp.exists(shadow_mode_path):
                os.makedirs(shadow_mode_path)
            shadow_log_path = osp.join(shadow_mode_path, '{}.log.tsv'.format(sm_m))

            s_checkpoint_suffix = '{}'.format(len(shadow_set))

            if cfg.read_shadow_from_path:
                shadow_model, _ = Blackbox.from_modeldir_split_attack_mode(shadow_mode_path, 'checkpoint_6000.pth.tar',device)

            else:
                if cfg.transfer_mi:
                    shadow_model = zoo.get_net(attack_model_name, modelfamily, num_classes=num_classes)
                    shadow_model = shadow_model.to(device)
                    optimizer1 = get_optimizer(shadow_model.parameters(), params['optimizer_choice'], **params)

                    transmi_path = osp.join(cfg.TRANSMI_DIR, 'checkpoint.pth.tar')
                    checkpoint_trans = torch.load(transmi_path)
                    # print("keys:", checkpoint_trans['state_dict'].keys())
                    # here we cant
                    for key in checkpoint_trans['state_dict'].keys():
                        if key.find('block3') == 0 or key.find("bn1") == 0 or key.find('fc') == 0:
                            print(key)
                            # covered by random
                            checkpoint_trans['state_dict'][key] = shadow_model.state_dict()[key]
                    shadow_model.load_state_dict(checkpoint_trans['state_dict'])
                    # checkpoint_suffix_s = '{}'.format(len(shadow_set))
                    print("start training shadow_model")
                    # *****************
                    model_utils.train_model(model=shadow_model, trainset=shadow_set, out_path=shadow_mode_path,
                                            blackbox=blackbox, testset=testset,
                                            criterion_train=criterion_train,
                                            checkpoint_suffix=s_checkpoint_suffix, device=device,
                                            optimizer=optimizer1,
                                            s_m=sm_m, args=model_args, **params)

                    p_shadow = argparse.ArgumentParser(description='Train a model')
                    args_shadow = p_shadow.parse_args()
                    p_save = vars(args_shadow)
                    p_save['model_arch'] = cfg.victim_model_arch
                    p_save['num_classes'] = num_classes
                    p_save['dataset'] = cfg.test_dataset
                    p_save['created_on'] = str(datetime.datetime.now())
                    p_save['start_point'] = str(cfg.start_shadow_test)
                    p_save['start_point_out'] = str(cfg.start_shadow_test_out)
                    p_save['query_dataset'] = str(cfg.queryset)
                    s_params_out_path = osp.join(shadow_mode_path, 'params.json')
                    with open(s_params_out_path, 'w') as jf:
                        json.dump(p_save, jf, indent=True)
                    print("start training shadow_attack_model")
                    shadow_model = Blackbox(shadow_model, device, 'probs')
                else:
                    num_classes = 10  # 先设一个，对mnist
                    p_shadow = argparse.ArgumentParser(description='Train a model')
                    args_shadow = p_shadow.parse_args()
                    p_save = vars(args_shadow)
                    p_save['model_arch'] = cfg.victim_model_arch
                    p_save['num_classes'] = num_classes
                    p_save['dataset'] = cfg.test_dataset

                    shadow_model = zoo.get_net(attack_model_name, modelfamily, num_classes=num_classes)
                    shadow_model = shadow_model.to(device)

                    optimizer2 = get_optimizer(shadow_model.parameters(), params['optimizer_choice'], **params)
                    # s_checkpoint_suffix = '{}'.format(len(shadow_set))
                    print("##############start training shadow_model#####################")
                    # ep_temp = params['epochs']
                    # params['epochs'] = 50
                    if not cfg.read_attack_mia_model_from_path:
                        model_utils.train_model(model=shadow_model, trainset=shadow_set, out_path=shadow_mode_path,
                                            blackbox=blackbox, testset=testset,
                                            criterion_train=criterion_train,
                                            checkpoint_suffix=s_checkpoint_suffix, device=device,
                                            optimizer=optimizer2,
                                            s_m=sm_m, args=model_args, **params)
                    # params['epochs'] = ep_temp
                    shadow_model = Blackbox(shadow_model, device, 'probs')

                    p_save['created_on'] = str(datetime.datetime.now())
                    p_save['start_point'] = str(cfg.start_shadow_test)
                    p_save['start_point_out'] = str(cfg.start_shadow_test_out)
                    p_save['query_dataset'] = str(cfg.queryset)
                    s_params_out_path = osp.join(shadow_mode_path, 'params.json')
                    with open(s_params_out_path, 'w') as jf:
                        json.dump(p_save, jf, indent=True)
                    print("start training shadow_attack_model")

            transferset_shadow_out = adversary.get_transferset_shadow_out(
                shadow_out_len=len(shadow_set))
            #out of shadow model's training dataset
            shadow_set_out = samples_to_transferset(transferset_shadow_out, budget=len(transferset_shadow_out),
                                                    transform=test_transform)
            if cfg.read_attack_mia_model_from_path:
                shadow_attack_model = load_bi_classifier(shadow_model=shadow_model, shadow_set=shadow_set_2,
                                                  shadow_set_out=shadow_set_out, device=device,
                                                  shadow_model_path=shadow_mode_path)
            else:
                shadow_attack_model = train_bi_classifier(shadow_model=shadow_model, shadow_set=shadow_set_2,
                                                      shadow_set_out=shadow_set_out, device=device, shadow_model_path=shadow_mode_path)  #we use not transfered data

            # Here test
            if cfg.mi_test:
                len_test = 20000
                y_mi_test, y_mi_target = s_adversary.get_transferset_shadow_test(k=len_test)
                confidence = []
                for input_batch, _ in classifier.iterate_minibatches(inputs=y_mi_test, targets=y_mi_target,
                                                                     batch_size=cfg.batch_size, shuffle=False):
                    # print("shadow_model_attack: input_batch:", input_batch.shape)
                    input = clipDataTopX(input_batch, top=3)
                    # top= [i[0]>0.5 for i in input]
                    # print("top", top)
                    pred = shadow_attack_model(input)  # output
                    confidence.append([p[1] for p in pred])
                confidence = np.concatenate(confidence)
                esti_training_50 = []
                esti_training_55 = []
                esti_training_60 = []
                esti_training_65 = []
                esti_no_training_50 = []
                esti_no_training_55 = []
                esti_no_training_60 = []
                esti_no_training_65 = []
                for idx, c in enumerate(confidence):
                    if c > 0.5:
                        esti_training_50.append(idx)
                    else:
                        esti_no_training_50.append(idx)
                    if c > 0.55:
                        esti_training_55.append(idx)
                    else:
                        esti_no_training_55.append(idx)
                    if c > 0.6:
                        esti_training_60.append(idx)
                    else:
                        esti_no_training_60.append(idx)
                    if c > 0.65:
                        esti_training_65.append(idx)
                    else:
                        esti_no_training_65.append(idx)
                print("threshold=0.5")
                tp1 = np.sum(np.asarray(esti_training_50) < len_test / 2)
                acc1 = np.sum(np.asarray(esti_training_50) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_50) >= len_test / 2)
                print("true_positive:", tp1, '/', len(esti_training_50))
                print("acc:", acc1 / len_test)

                print("threshold=0.55")
                tp2 = np.sum(np.asarray(esti_training_55) < len_test / 2)
                acc2 = np.sum(np.asarray(esti_training_55) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_55) >= len_test / 2)
                print("true_positive:", tp2, '/', len(esti_training_55))
                print("acc:", acc2 / len_test)

                print("threshold=0.6")
                tp3 = np.sum(np.asarray(esti_training_60) < len_test / 2)
                acc3 = np.sum(np.asarray(esti_training_60) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_60) >= len_test / 2)
                print("true_positive:", tp3, '/', len(esti_training_60))
                print("acc:", acc3 / len_test)

                print("threshold=0.65")
                tp4 = np.sum(np.asarray(esti_training_65) < len_test / 2)
                acc4 = np.sum(np.asarray(esti_training_65) < len_test / 2) + np.sum(
                    np.asarray(esti_no_training_65) >= len_test / 2)
                print("true_positive:", tp4, '/', len(esti_training_65))
                print("acc:", acc4 / len_test)

                with open(shadow_log_path, 'a') as af:
                    columns = ['shadow_model_acc_len', 'acc0.5','tp','esti_all', 'acc0.55', 'tp','esti_all', 'acc0.6','tp','esti_all', 'acc0.65', 'tp','esti_all']
                    af.write('\t'.join(columns) + '\n')
                    train_cols = [len(shadow_set), acc1 / len_test, tp1, esti_training_50,
                                  acc2 / len_test, tp2, esti_training_55, acc3 / len_test, tp3, esti_training_60,
                                  acc4 / len_test, tp4, esti_training_60]
                    af.write('\t'.join([str(c) for c in train_cols]) + '\n')

        #Pre-Filter:
        if cfg.imp_vic_mem:
            transferset_o = adversary.mia_on_query_result(mi_attacker=shadow_attack_model)
            transferset = samples_to_transferset(transferset_o, budget=len(transferset_o), transform=transform_query)

        # -----initial
        # torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)  # 使 model的初始化方式一样
        b = 0
        out_path = osp.join(attack_model_dir, cfg.queryset)
        if not osp.exists(out_path):
            os.makedirs(out_path)

        shadow_idx_path = osp.join(out_path, 'shadow_size-{}.tsv'.format(sm_m))
        shadow_idx_head = ['epoch', 'shadow_index_size']
        with open(shadow_idx_path, 'a') as af:  # overwriten
            af.write('\t'.join(shadow_idx_head) + '\n')
        if params['iterative']:  # start iteration train
            for it in range(params['num_iter']):
                it_time = []
                print('\n---------------start {} iteration'.format(it))
                # ----- process data
                if it == 0:
                    b = b + params['initial_seed']
                else:
                    b = b + params['k']

                # ----- restart attack_model
                attack_model = zoo.get_net(attack_model_name, modelfamily, pretrained, num_classes=num_classes)
                attack_model = attack_model.to(device)
                ema_model = zoo.get_net(attack_model_name, modelfamily, pretrained, num_classes=num_classes)
                ema_model = ema_model.to(device)

                for param in ema_model.parameters():
                    param.detach_()

                optimizer = get_optimizer(attack_model.parameters(), params['optimizer_choice'], **params)

                checkpoint_suffix = '{}'.format(b)

                # ----- Choose next bunch of queried data
                if it < params['num_iter']:
                    attack_model_adv, _ = Blackbox.from_modeldir_split_attack_mode(out_path,
                                                                               'checkpoint_{}.pth.tar'.format(
                                                                                   checkpoint_suffix), device)
                    adversary.set_attack_model(attack_model_adv, device)  # 将attack model输入
                    if params['sampling_method'] == 'membership_attack':  # membership
                        print("params['ma-method']:", params['ma_method'])
                        if params['ma_method'] == 'shadow_model':
                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_,
                                                                                   shadow_attack_model=shadow_attack_model,
                                                                                   second_sss=params['second_sss'],
                                                                                   it=it)
                        elif params['ma_method'] == 'unsupervised':
                            print("update unsuperY")
                            adversary.set_unsuperY()

                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_)
                        else:
                            transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'],
                                                                                   sampling_method=params[
                                                                                       'sampling_method'],
                                                                                   ma_method=params['ma_method'],
                                                                                   pre_idxset=pre_idxset_)
                    else:  # kcenter; adaptive; adversarial
                        transferset_o, pre_idxset_ = adversary.get_transferset(k=params['k'], sampling_method=params[
                                                                               'sampling_method'],
                                                                               ma_method=params['ma_method'],
                                                                               pre_idxset=pre_idxset_,
                                                                               shadow_attack_model=shadow_attack_model,
                                                                               second_sss=params['second_sss'])  # 不去管ma_method

                    print("choose_finished: transformset_o:", len(transferset_o))
                    # change_to_trainable_set
                    transferset = samples_to_transferset(transferset_o, budget=len(transferset_o),
                                                         transform=transform_query)
                    print('=> Training at budget = {}'.format(len(transferset)))
                    # update(shadow_model)
                    it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #6

                    setting_path = osp.join(cfg.transfer_set_out_dir, cfg.queryset)
                    if not os.path.exists(setting_path):
                        os.makedirs(setting_path)
                    np.save(setting_path + "//" + sm_m, pre_idxset_)

                    it_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  #7

                    # vmi:
                    if cfg.imp_vic_mem:
                        transferset_o = adversary.mia_on_query_result(mi_attacker=shadow_attack_model)
                        transferset = samples_to_transferset(transferset_o, budget=len(transferset_o),
                                                             transform=transform_query)

                    unlabeled_transferset = adversary.get_unlabled_transferset()
                    if len(unlabeled_transferset) > 0:
                        unlabeled_set = samples_to_transferset(unlabeled_transferset, budget=len(unlabeled_transferset),
                                                               transform=TransformTwice(transform_query))  # transform_query
                    else:
                        unlabeled_set = None

                    if unlabeled_set is not None:
                        print("unlabeled_set is not None")
                        model_utils_semi.train_model(model=attack_model, ema_model=ema_model, trainset=transferset,
                                                     unlabeled_set=unlabeled_set, out_path=out_path,
                                                     blackbox=blackbox,
                                                     testset=testset,
                                                     criterion_train=criterion_train,
                                                     checkpoint_suffix=checkpoint_suffix, device=device,
                                                     optimizer=optimizer,
                                                     s_m=sm_m, args=model_args, **params)
                    else:
                        print("do nothing")

            print("pre+idexset", pre_idxset_)  # 显示data的idx
        else:
            print("No implemented")

if __name__ == '__main__':
    main()
