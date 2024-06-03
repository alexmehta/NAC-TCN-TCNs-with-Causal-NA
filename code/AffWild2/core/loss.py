"""
Author: HuynhVanThong
"""
import torch
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
from torch import nn


def CCCLoss(y_hat, y, scale_factor=1., num_classes=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes))

    yhat_mean = torch.mean(y_hat_fl, dim=0, keepdim=True)
    y_mean = torch.mean(y_fl, dim=0, keepdim=True)

    sxy = torch.mean(torch.mul(y_fl - y_mean, y_hat_fl - yhat_mean), dim=0)
    rhoc = torch.div(2 * sxy,
                     torch.var(y_fl, dim=0) + torch.var(y_hat_fl, dim=0) + torch.square(y_mean - yhat_mean) + 1e-8)

    return 1 - torch.mean(rhoc)


def MSELoss(y_hat, y, scale_factor=1., num_classes=2):
    mse_loss = torch.mean(torch.square(y_hat - y * scale_factor), dim=0)
    return torch.mean(mse_loss)


def BCEwithLogitsLoss(y_hat, y, scale_factor=1, num_classes=12, pos_weights=None):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes)) * 1.0

    return F.binary_cross_entropy_with_logits(y_hat_fl, y_fl, reduction='mean', pos_weight=pos_weights)


def SigmoidFocalLoss(y_hat, y, scale_factor=1, num_classes=12, pos_weights=None, alpha=0.25, gamma=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes)) * 1.0
    loss_val = sigmoid_focal_loss(
        inputs=y_hat_fl, targets=y_fl, alpha=alpha, gamma=gamma, reduction='mean')
    return loss_val


def CELogitLoss(y_hat, y, scale_factor=1, num_classes=8, label_smoothing=0., class_weights=None):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1,))  # Class indices

    return F.cross_entropy(y_hat_fl, y_fl, reduction='mean', weight=class_weights, label_smoothing=label_smoothing)


def CEFocalLoss(y_hat, y, scale_factor=1, num_classes=8, label_smoothing=0., class_weights=None, alpha=0.25, gamma=2.,
                distillation_loss=True):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1,))  # Class indices

    ce_loss = F.cross_entropy(
        y_hat_fl, y_fl, label_smoothing=label_smoothing, reduction='none')
    target_one_hot = F.one_hot(y_fl, num_classes=num_classes)
    p = F.softmax(y_hat_fl, dim=1)

    if distillation_loss:
        target_one_hot_smooth = target_one_hot * \
            (1 - label_smoothing) + label_smoothing / num_classes
        dist_loss = F.kl_div(F.log_softmax(y_hat_fl, dim=1),
                             target_one_hot_smooth, reduction='batchmean')
    else:
        dist_loss = 0.

    # + (1 - p) * (1 - target_one_hot)
    p_t = torch.sum(p * target_one_hot, dim=1)
    loss = ce_loss * torch.pow(1 - p_t, gamma)
    if alpha > 0.:
        alpha_t = torch.sum(alpha * target_one_hot, dim=1)
        loss = alpha_t * loss

    if distillation_loss:
        dist_loss_coeff = 0.2
        return loss.mean() * (1 - dist_loss_coeff) + dist_loss_coeff * dist_loss
    else:
        return loss.mean()
