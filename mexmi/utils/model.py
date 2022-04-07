#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import time
from datetime import datetime
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as torch_models

import mexmi.config as cfg
import mexmi.utils.utils as attack_utils

def get_net(model_name, n_output_classes=1000, **kwargs):
    print('=> loading model {} with arguments: {}'.format(model_name, kwargs))
    valid_models = [x for x in torch_models.__dict__.keys() if not x.startswith('__')]
    if model_name not in valid_models:
        raise ValueError('Model not found. Valid arguments = {}...'.format(valid_models))
    model = torch_models.__dict__[model_name](**kwargs)
    # Edit last FC layer to include n_output_classes
    if n_output_classes != 1000:
        if 'squeeze' in model_name:
            model.num_classes = n_output_classes
            model.classifier[1] = nn.Conv2d(512, n_output_classes, kernel_size=(1, 1))
        elif 'alexnet' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'vgg' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, n_output_classes)
        elif 'dense' in model_name:
            model.num_classes = n_output_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, n_output_classes)
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, n_output_classes)
    return model


def soft_cross_entropy(pred, soft_targets, weights=None):
    if weights is not None:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
    else:
        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, train_gt_loader=None, criterion=None, optimizer=None, epoch=None, device=None, log_interval=20, writer=None):
    model.train()
    train_loss = 0.
    correct = 0
    total = 0
    train_loss_batch = 0
    epoch_size = len(train_loader.dataset)
    t_start = time.time()
    # print("epoch size:", len(train_loader.dataset))
    #
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # print("targets in training step,", targets)
        if train_gt_loader is not None:
            (_, gt_labels) = train_gt_loader.__iter__().__next__()
            # print("gt_labels in training step,", gt_labels)
            loss2 = criterion(outputs, gt_labels.to(device))
            loss = 0.5*loss + 0.5*loss2
        loss.backward()
        optimizer.step()
        if writer is not None:
            pass

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if len(targets.size()) == 2:
            # Labels could be a posterior probability distribution. Use argmax as a proxy.
            target_probs, target_labels = targets.max(1)
        else:
            target_labels = targets
        correct += predicted.eq(target_labels).sum().item()

        prog = total / epoch_size
        exact_epoch = epoch + prog - 1
        acc = 100. * correct / total
        train_loss_batch = train_loss / total

        if (batch_idx + 1) % log_interval == 0:
            print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
                exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), acc, correct, total))

        if writer is not None:
            writer.add_scalar('Loss/train', loss.item(), exact_epoch)
            writer.add_scalar('Accuracy/train', acc, exact_epoch)

    t_end = time.time()
    t_epoch = int(t_end - t_start)
    acc = 100. * correct / total

    return train_loss_batch, acc


def test_step(model, test_loader, criterion, device, epoch=0., blackbox=None, silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    total = 0
    fid_num = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if blackbox is not None:
                # print("calculate fidelity")
                truel = blackbox(inputs)
                _, true_label =truel.max(1)
                fid_num += predicted.eq(true_label).sum().item()

            # if batch_idx >= 1249:
            #     break

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    fidelity = 100. * fid_num / total
    test_loss /= total

    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\t Fidelity:{}'.format(epoch, test_loss, acc,
                                                                             correct, total, fidelity))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)

    return test_loss, acc, fidelity


def train_model(model, trainset, trainset_gt=None, out_path=None, blackbox=None, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, s_m=None, **kwargs):
    print('train_model_function')
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        attack_utils.create_dir(out_path)
    run_id = str(datetime.now())

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    if trainset_gt is not None:
        train_gt_loader = DataLoader(trainset_gt, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_gt_loader = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    #How
    if weighted_loss:#loss
        if not isinstance(trainset.samples[0][1], int):
            print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

        class_to_count = dd(int)
        for _, y in trainset.samples:
            class_to_count[y] += 1
        class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
        print('=> counts per class: ', class_sample_count)
        weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
        weight = weight.to(device)
        print('=> using weights: ', weight)
    else:
        weight = None

    # Optimizer
    if criterion_train is None:
        criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss, best_fidelity= -1., -1., -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    # Initialize logging
    # log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
    log_path = osp.join(out_path, '{}.log.tsv'.format(s_m))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            # columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            columns = ['run_id', 'loss', 'epochs', 'query_number', 'training_acc', 'test_acc', 'fidelity']
            wf.write('\t'.join(columns) + '\n')
    # with open(log_path, 'a') as wf:
    #     columns = [s_m, "","","","",""]
    #     wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1): #1，101
        
        train_loss, train_acc = train_step(model, train_loader, train_gt_loader, criterion_train, optimizer, epoch, device, log_interval
                                           )#log_interval=log_interval
        # scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if (epoch + 100) >= epochs:
            if test_loader is not None:
                test_loss, test_acc, test_fidelity = test_step(model, test_loader, criterion_test, device, epoch=epoch,
                                                               blackbox=blackbox)
                best_test_acc = max(best_test_acc, test_acc)
                best_fidelity = max(best_fidelity, test_fidelity)

            # Checkpoint
            if test_acc >= best_test_acc:
                state = {
                    'epoch': epoch,
                    'arch': model.__class__,
                    'state_dict': model.state_dict(),
                    'best_acc': test_acc,
                    'optimizer': optimizer.state_dict(),
                    'created_on': str(datetime.now()),
                }
                torch.save(state, model_out_path)

        # Log
    #columns = ['run_id', 'loss', 'query_number', 'training_acc', 'test_acc', 'fidelity']
    with open(log_path, 'a') as af:
        train_cols = [run_id, train_loss, epochs, checkpoint_suffix, best_train_acc, best_test_acc, best_fidelity]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
        # af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return model
