#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset

import mexmi.config as cfg
import parser_params
from blackbox import Blackbox
from mexmi import datasets
import mexmi.utils.transforms as transform_utils
import mexmi.utils.model_scheduler as model_utils
import mexmi.utils.utils as mexmi_utils
import mexmi.models.zoo as zoo

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    #arguments, not required now
    parser.add_argument('--dataset', metavar='DS_NAME', type=str, help='Dataset name', default='CIFAR10') #CIFAR10
    parser.add_argument('--model_arch', metavar='MODEL_ARCH', type=str, help='Model name', default='wide_resnet28_10')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.VICTIM_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)') #0.1
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None) #None cfg.PRETRAIN_DIR
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)
    parser.add_argument('--work_mode', action='store_true', help='Use a weighted loss', default='train_vicitm')
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    print("valid_datasets:", valid_datasets)
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    print("modelfamily", modelfamily)

    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    print("train_transform:", train_transform.__class__)
    #
    trainset = dataset(train=True, transform=train_transform) #, download=True
    testset = dataset(train=False, transform=test_transform) #, download=True

    num_classes = 10#len(trainset.classes)
    params['num_classes'] = num_classes

    if params['train_subset'] is not None:
        idxs = np.arange(len(trainset))
        ntrainsubset = params['train_subset']
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        trainset = Subset(trainset, idxs)

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    # pretrained = Blackbox.from_modeldir(params['pretrained'], device)
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    print("model", model)
    model = model.to(device)

    model_parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
    model_args = parser_params.add_parser_params(model_parser)

    # ----------- Train
    out_path = params['out_path']
    model_utils.train_model(model=model, trainset=trainset, testset=testset, device=device, args=model_args, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
