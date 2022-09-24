import numpy as np
import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        probas = pred.softmax(dim=1)
        loss = -(target*((1-probas)**self.gamma)*(probas.log())).mean()
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, LossFunction: str, num_task, weight=None):
        super().__init__()
        self.num_task = num_task
        if weight is None:
            self.weight = np.array([1.0] * self.num_task)
        else:
            assert(len(weight)==num_task)
            self.weight = np.array(weight)
        self.norm_weight = torch.tensor(self.weight / np.sum(self.weight)).cuda()
        self.criterion = eval(LossFunction)()

    def forward(self, pred, target):
        loss = torch.tensor(0).float().cuda()
        for i in range(self.num_task):
            loss += self.norm_weight[i] * self.criterion(pred[:, i], target[:, i])
        return loss



class PseudoLoss(nn.Module):
    def __init__(self, LossFunction: str, alpha=3, epoch_th_lower=100, epoch_th_upper=600):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.epoch_th_lower = torch.tensor(epoch_th_lower)
        self.epoch_th_upper = torch.tensor(epoch_th_upper)
        self.weight_norm_factor = self.epoch_th_upper - self.epoch_th_lower
        self.loss = eval(LossFunction)(reduction="none")

    def forward(self, pred, target, pseudo, epoch):
        gt_loss = self._calc_ground_truth_label_loss(pred, target, pseudo)
        pseudo_loss = self._calc_pseudo_label_loss(pred, target, pseudo, epoch)
        return gt_loss + pseudo_loss

    def _calc_ground_truth_label_loss(self, pred, target, pseudo):
        if pseudo.all():
            return torch.tensor(0.0)
        gt_label_flag = (~pseudo).flatten().to(torch.int)
        return (self.loss(pred, target) * gt_label_flag).sum() / (~pseudo).sum()

    def _calc_pseudo_label_loss(self, pred, target, pseudo, epoch):
        if (~pseudo).all():
            return torch.tensor(0.0)
        pseudo_loss_weight = self._calc_psuedo_loss_weight(epoch)
        pseudo_label_flag = pseudo.flatten().to(torch.int)
        return pseudo_loss_weight * (self.loss(pred, target) * pseudo_label_flag).sum() / pseudo.sum()

    def _calc_psuedo_loss_weight(self, epoch):
        if epoch < self.epoch_th_lower:
            return torch.tensor(0.0)
        if epoch >= self.epoch_th_upper:
            return self.alpha
        return self.alpha * (epoch - self.epoch_th_lower) / self.weight_norm_factor
