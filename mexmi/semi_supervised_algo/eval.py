from __future__ import print_function, absolute_import

__all__ = ['train_accuracy']

import torch


def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    _, target = torch.max(target, dim=-1)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # view --> reshape
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    #
    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    # return res