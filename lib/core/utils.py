import os
import numpy as np
import shutil
import torch
from sklearn.metrics import average_precision_score


def str_gpus(ids):
    str_ids = ''
    for id in ids:
        str_ids = str_ids + str(id)
        str_ids = str_ids + ','

    return str_ids


def map_sklearn(labels, results):
    map = average_precision_score(labels, results, average="micro")
    return map


def adjust_learning_rate(optimizer, epoch, cfg):
    """"Sets the learning rate to the initial LR decayed by lr_factor"""
    lr_decay = cfg.SOLVER.LR_FACTOR**(sum(epoch > np.array(cfg.SOLVER.LR_STEPS)))
    lr = cfg.SOLVER.START_LR * lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']


def save_checkpoint(state, save_dir, epoch, is_best):
    filename = os.path.join(save_dir, 'ckpt_'+str(epoch)+'.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def list2acc(results_list):
    """
    :param results_list: list contains 0 and 1
    :return: accuarcy
    """
    accuarcy = results_list.count(1)/len(results_list)
    return accuarcy