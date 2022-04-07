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
import mexmi.utils.utils as mexmi_utils

import lr_scheduler
from semi_supervised_algo import Bar, Logger, AverageMeter, train_accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--semi_epochs', default=32, type=int, metavar='N',
                    help='number of total epochs to run(1024)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data')
# parser.add_argument('--train-iteration', type=int, default=1024,
#                         help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--train_iteration', type=int, default=512,
                        help='Number of iteration per epoch1024')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


semi_args = parser.parse_args()
state = {k: v for k, v in semi_args._get_kwargs()}

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * semi_args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

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

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def train_step(model, train_loader,unlabeled_loader, train_gt_loader=None, criterion=None, optimizer=None, ema_optimizer=None,epoch=None, device=None, log_interval=20, scheduler=None, writer=None):
    # model.train()
    # train_loss = 0.
    # correct = 0
    # total = 0
    # train_loss_batch = 0
    # epoch_size = len(train_loader.dataset)
    # t_start = time.time()

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()
    semi_args.train_iteration = len(train_loader)*3
    # bar = Bar('Training', max=semi_args.train_iteration)
    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabeled_loader)

    
    i=0
    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx in range(semi_args.train_iteration):
        # print("batch_idx", batch_idx)
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            print("in except inputs_x")
            labeled_train_iter = iter(train_loader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            print("in except inputs_u")
            unlabeled_train_iter = iter(unlabeled_loader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        # inputs_x, targets_x = labeled_train_iter.next()

        # inputs, targets = inputs.to(device), targets.to(device)
        scheduler(optimizer, i, epoch)
        i += 1

        b_size = inputs_x.size(0)

        inputs_x, targets_x = inputs_x.to(device), targets_x.to(device) #targets is one-hot
        inputs_u = inputs_u.to(device)
        inputs_u2 = inputs_u2.to(device)

        # optimizer.zero_grad()
        # ema_optimizer.zero_grad()
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / semi_args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            weight_u = torch.ones(targets_u.shape)
            weight_u = weight_u.to(device)
            targets_u = torch.stack((targets_u, weight_u), dim=1)

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(semi_args.alpha, semi_args.alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # print("0batch_size,", b_size)
        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, b_size))
        mixed_input = interleave(mixed_input, b_size)
        # print('0mixed_input,', mixed_input[0].shape)
        logits = [model(mixed_input[0])]
        # print('1logits,', logits[0].shape)
        for input in mixed_input[1:]:
            logits.append(model(input))

        # print('2logits,', logits[0].shape)
        # print("1batch_size,", b_size)
        # put interleaved samples back
        logits = interleave(logits, b_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # print('3logits,', logits[0].shape)
        # b_size = logits_x.size(0)
        # print("2batch_size,", b_size,',logits_x,',logits_x.shape, ',mixed_target,', mixed_target.shape)
        b_size = logits_x.size(0)
        Lx, Lu, w = criterion(logits_x, mixed_target[:b_size], logits_u, mixed_target[b_size:],
                              epoch + batch_idx / semi_args.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # prog = total / epoch_size
        # exact_epoch = epoch + prog - 1

        # plot progress
        # bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
        #     batch=batch_idx + 1,
        #     size=len(train_loader),
        #     data=data_time.avg,
        #     bt=batch_time.avg,
        #     total=bar.elapsed_td,
        #     eta=bar.eta_td,
        #     loss=losses.avg,
        #     loss_x=losses_x.avg,
        #     loss_u=losses_u.avg,
        #     w=ws.avg,
        # )
        # bar.next()
        if (batch_idx + 1) % log_interval == 0:
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                batch=batch_idx + 1,
                size=len(train_loader)*3,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                w=ws.avg,
            ))
    # bar.finish()


    return (losses.avg, losses_x.avg, losses_u.avg,)


def test_step(model, test_loader, criterion, device, epoch=0., blackbox=None, blackbox_test_result=None, batch_size=32, silent=False, writer=None):
    model.eval()
    test_loss = 0.
    correct = 0
    correct_top5 = 0
    total = 0
    fid_num = 0
    fid_num5 = 0

    t_start = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # nclasses = outputs.size(1)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, pred5 = outputs.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct5 = pred5.eq(targets.view(1, -1).expand_as(pred5))
            correct_top5 += correct5[:5].reshape(-1).float().sum(0, keepdim=True)  # view --> reshape

            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
            if blackbox is not None:
                truel = blackbox(inputs)
                _, true_label =truel.max(1)
                fid_num5_t = pred5.eq(true_label.view(1, -1).expand_as(pred5))
                fid_num5 += fid_num5_t[:5].reshape(-1).float().sum(0, keepdim=True)
                fid_num += predicted.eq(true_label).sum().item()
            elif blackbox_test_result is not None:
                # truel = blackbox(inputs)
                truel = blackbox_test_result[batch_idx*batch_size:min((batch_idx+1)*batch_size, len(blackbox_test_result))]
                _, true_label =torch.tensor(truel).to(device).max(1)
                fid_num5_t = pred5.eq(true_label.view(1, -1).expand_as(pred5))
                fid_num5 += fid_num5_t[:5].reshape(-1).float().sum(0, keepdim=True)
                fid_num += predicted.eq(true_label).sum().item()
                # print("blackbox_test_result is not None")

            # if batch_idx >= 1249: 
            #     break

    t_end = time.time()
    t_epoch = int(t_end - t_start)

    acc = 100. * correct / total
    fidelity = 100. * fid_num / total
    fidelity5 = 100. * fid_num5 / total
    test_loss /= total
    acc5 = 100. * correct_top5 / total


    if not silent:
        print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})\t Fidelity:{}'.format(epoch, test_loss, acc,
                                                                             correct, total, fidelity))

    if writer is not None:
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', acc, epoch)
        writer.add_scalar('Fidelity/test', fidelity, epoch)

    return test_loss, acc, acc5.cpu().numpy()[0], fidelity, fidelity5.cpu().numpy()[0]

def test_step2(model, thir_model, four_model, fif_model, six_model, test_loader, criterion, device, epoch=0.,
               blackbox=None, silent=False, writer=None):
    # model.eval()
    # thir_model.eval()
    # four_model.eval()

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
            if thir_model is not None:
                outputs2 = thir_model(inputs)
            else:
                outputs2 = torch.zeros([10]).to(device)
            if four_model is not None:
                outputs3 = four_model(inputs)
            else:
                outputs3 = torch.zeros([10]).to(device)
            if fif_model is not None:
                outputs4 = fif_model(inputs)
            else:
                outputs4 = torch.zeros([10]).to(device)
            if six_model is not None:
                outputs5 = six_model(inputs)
            else:
                outputs5 = torch.zeros([10]).to(device)
            test_loss += loss.item()
            _, predicted = (outputs + outputs2 + outputs3 + outputs4 +outputs5).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if blackbox is not None:
                truel = blackbox(inputs)
                _, true_label =truel.max(1)
                fid_num += predicted.eq(true_label).sum().item()

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

def test_model(blackbox=None, blackbox2=None, blackbox3=None, blackbox4=None, blackbox5=None, blackbox6=None,
               batch_size=10, testset=None,
               num_workers=10, criterion_test=None, device=None,
               epoch=100, **kwangs):
    weight = None
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_loader = None
    if criterion_test is None:
        criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)

    if test_loader is not None:
        test_loss, test_acc, test_fidelity = test_step2(model=blackbox2, thir_model=blackbox3, four_model=blackbox4,
                                                        fif_model=blackbox5, six_model=blackbox6, test_loader=test_loader,
                                                       device=device, epoch=epoch, criterion=criterion_test,
                                                       blackbox=blackbox)

def linear_rampup(current, rampup_length=cfg.epoch):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x[:, 0] * targets_x[:, 1], dim=1))
        Lu = torch.mean(((probs_u - targets_u[:,0])*targets_u[:, 1]) **2)

        return Lx, Lu, semi_args.lambda_u * linear_rampup(epoch)

def train_validate(valloader, model, criterion, epoch, device, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets[:, 0].to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = train_accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def train_model(model, ema_model, trainset=None, unlabeled_set=None, trainset_gt=None, out_path=None, blackbox=None, blackbox_test_result=None, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
                device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
                epochs=100, log_interval=10, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
                writer=None, s_m=None, args=None, **kwargs):
    #change optimizer
    param_groups = model.parameters() if args.is_wd_all else lr_scheduler.get_parameter_groups(model)
    # if args.optimizer == 'SGD':
    print("INFO:PyTorch: using SGD optimizer.")
    #change the optimizer directly
    optimizer = torch.optim.SGD(param_groups,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True
                                )
    print('train_model_function')
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        mexmi_utils.create_dir(out_path)
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

    if unlabeled_set is not None:
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        unlabeled_loader = None

    # Optimizer
    # for semi-supervised
    ema_optimizer = WeightEMA(model, ema_model, alpha=semi_args.ema_decay)

    # learning rate scheduler
    scheduler = lr_scheduler.lr_scheduler(mode=args.lr_mode,
                                          init_lr=args.lr,
                                          num_epochs=args.epochs,
                                          iters_per_epoch=len(train_loader),
                                          lr_milestones=args.lr_milestones,
                                          lr_step_multiplier=args.lr_step_multiplier,
                                          slow_start_epochs=args.slow_start_epochs,
                                          slow_start_lr=args.slow_start_lr,
                                          end_lr=args.end_lr,
                                          multiplier=args.lr_multiplier,
                                          decay_factor=args.decay_factor,
                                          decay_epochs=args.decay_epochs,
                                          staircase=True
                                          )

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
    # if optimizer is None:
    #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    # if scheduler is None:
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_train_acc, train_acc = -1., -1.
    best_test_acc, test_acc, test_loss, best_fidelity= -1., -1., -1., -1.
    # Resume if required 从某个模型继续训练
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
    # log_path = osp.join(out_path, 'checkpoint_{}.log.tsv'.format(checkpoint_suffix))
    log_path = osp.join(out_path, '{}.log.tsv'.format(s_m))
    if not osp.exists(log_path):
        with open(log_path, 'w') as wf:
            # columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
            columns = ['run_id', 'loss', 'epochs', 'query_number', 'training_acc', 'best_training_acc','test_acc@1', 'test_acc@5', 'fidelity@1', 'fidelity@5']
            wf.write('\t'.join(columns) + '\n')
    # with open(log_path, 'a') as wf:
    #     columns = [s_m, "","","","",""]
    #     wf.write('\t'.join(columns) + '\n')

    model_out_path = osp.join(out_path, 'checkpoint_{}.pth.tar'.format(checkpoint_suffix))
    # model_out_path = osp.join(out_path, '{}_50000.pth.tar'.format(s_m))
    semi_criterion = SemiLoss()

    for epoch in range(start_epoch, epochs + 1): #1，101
        #
        train_loss, train_loss_x, train_loss_u = train_step(model, train_loader, unlabeled_loader=unlabeled_loader,
                                                            train_gt_loader=train_gt_loader, criterion=semi_criterion,
                                                            optimizer=optimizer, ema_optimizer=ema_optimizer,
                                                            epoch=epoch, device=device, log_interval=log_interval,
                                                            scheduler=scheduler)#log_interval=log_interval

        _, train_acc = train_validate(train_loader, ema_model, criterion_train, epoch, device, mode='Train Stats')

        # scheduler.step(epoch)
        best_train_acc = max(best_train_acc, train_acc)

        if True:#(epoch+10) >=epochs:
            if test_loader is not None:
                test_loss, test_acc, test_acc5, test_fidelity, test_fidelity5 = test_step(ema_model, test_loader, criterion_test, device, epoch=epoch,
                                                               blackbox=blackbox, blackbox_test_result=blackbox_test_result, batch_size=batch_size)
                is_best = (best_fidelity < test_fidelity)
                if is_best:
                    best_test_acc = test_acc
                    best_fidelity = test_fidelity
                    best_test_acc5 = test_acc5
                    best_fidelity5 = test_fidelity5

                    # Checkpoint
                    # if test_acc >= best_test_acc:
                    state = {
                            'epoch': epoch,
                            'arch': ema_model.__class__,
                            'state_dict': ema_model.state_dict(),
                            'original_state_dict': model.state_dict(),
                            'best_acc': test_acc,
                            'optimizer': optimizer.state_dict(),
                            'created_on': str(datetime.now()),
                    }
                    torch.save(state, model_out_path)
        # Log
        if epoch % 5 == 0:
            with open(log_path, 'a') as af:
                train_cols = [run_id, train_loss, epoch, checkpoint_suffix, train_acc, best_train_acc, best_test_acc, best_test_acc5,
                      best_fidelity, best_fidelity5]
                af.write('\t'.join([str(c) for c in train_cols]) + '\n')
                # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
                # af.write('\t'.join([str(c) for c in test_cols]) + '\n')
    #columns = ['run_id', 'loss', 'query_number', 'training_acc', 'test_acc', 'fidelity']
    with open(log_path, 'a') as af:
        train_cols = [run_id, train_loss, epoch, checkpoint_suffix, train_acc,best_train_acc, best_test_acc, best_test_acc5,
                      best_fidelity, best_fidelity5]
        af.write('\t'.join([str(c) for c in train_cols]) + '\n')
        # test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc, test_fidelity]
        # af.write('\t'.join([str(c) for c in test_cols]) + '\n')

    return ema_model
