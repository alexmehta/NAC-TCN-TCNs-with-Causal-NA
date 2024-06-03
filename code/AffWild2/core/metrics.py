"""Author: HuynhVanThong"""
import torch
from typing import Tuple
from torch import Tensor
import torchmetrics
from torchmetrics.regression.pearson import _pearson_corrcoef_update, _pearson_corrcoef_compute
from torchmetrics.utilities.checks import _check_same_shape


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    corrs_xy: Tensor,
    nbs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Aggregate the statistics from multiple devices`_
    """
    # assert len(means_x) > 1 and len(means_y) > 1 and len(vars_x) > 1 and len(vars_y) > 1 and len(corrs_xy) > 1
    mean_x, mean_y, var_x, var_y, corr_xy, nb = means_x[
        0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]
        # vx2: batch_p_var, vy2: batch_l_var , cxy2: batch_bl_var
        delta_p_var = (vx2 + (mean_x - mx2) * (mean_x - mx2) * (
            nb * n2 / (nb + n2)))
        var_x += delta_p_var

        delta_l_var = (vy2 + (mean_y - my2) * (mean_y - my2) * (
            nb * n2 / (nb + n2)))
        var_y += delta_l_var

        delta_pl_var = (cxy2 + (mean_x - mx2)*(mean_y - my2) * (nb*n2)/(nb+n2))
        corr_xy += delta_pl_var

        nb += n2
        mean_x = (nb * mean_x + n2 * mx2) / nb
        mean_y = (nb * mean_y + n2 * my2) / nb

    return var_x, var_y, corr_xy, nb


def _corrcoeff_update(preds: Tensor,
                      target: Tensor,
                      mean_x: Tensor,
                      mean_y: Tensor,
                      var_x: Tensor,
                      var_y: Tensor,
                      corr_xy: Tensor,
                      n_prior: Tensor,
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Updates and returns variables required to compute Pearson Correlation Coefficient. Checks for same shape of
    input tensors.

    Args:
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        n_prior: current number of observed observations
    """
    # Data checking
    _check_same_shape(preds, target)

    preds = preds.squeeze()
    target = target.squeeze()
    if preds.ndim > 1 or target.ndim > 1:
        raise ValueError(
            "Expected both predictions and target to be 1 dimensional tensors.")

    n_obs = preds.numel()
    batch_mean_preds = preds.mean()
    batch_mean_labels = target.mean()

    mx_new = (n_prior * mean_x + preds.mean() * n_obs) / (n_prior + n_obs)
    my_new = (n_prior * mean_y + target.mean() * n_obs) / (n_prior + n_obs)

    batch_p_var = ((preds - batch_mean_preds) *
                   (preds - batch_mean_preds)).sum()
    delta_p_var = (batch_p_var + (mean_x - batch_mean_preds) * (mean_x - batch_mean_preds) * (
        n_prior * n_obs / (n_prior + n_obs)))
    var_x += delta_p_var

    batch_l_var = ((target - batch_mean_labels) *
                   (target - batch_mean_labels)).sum()
    delta_l_var = (batch_l_var + (mean_y - batch_mean_labels) * (mean_y - batch_mean_labels) * (
        n_prior * n_obs / (n_prior + n_obs)))
    var_y += delta_l_var

    batch_pl_corr = ((preds - batch_mean_preds) *
                     (target - batch_mean_labels)).sum()
    delta_pl_corr = (batch_pl_corr + (mean_x - batch_mean_preds) * (mean_y - batch_mean_labels) * (
        n_prior * n_obs / (n_prior + n_obs)))
    corr_xy += delta_pl_corr

    n_prior += n_obs

    mean_x = mx_new
    mean_y = my_new

    return mean_x, mean_y, var_x, var_y, corr_xy, n_prior


class ConCorrCoef(torchmetrics.Metric):
    """
    Based on: https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/regression/pearson.py
    """

    def __init__(self, num_classes=2, compute_on_step=False, dist_sync_on_step=False, process_group=None, ):
        super(ConCorrCoef, self).__init__()
        self.add_state('mean_x', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)
        self.add_state('mean_y', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)
        self.add_state('var_x', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)
        self.add_state('var_y', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)
        self.add_state('corr_xy', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)
        self.add_state('n_total', default=torch.tensor(
            [0.] * num_classes), dist_reduce_fx=None)

        self.num_classes = num_classes

    def update(self, yhat, y):
        preds = torch.reshape(yhat, (-1, self.num_classes))
        target = torch.reshape(y, (-1, self.num_classes))

        for idx in range(self.num_classes):
            self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx], self.corr_xy[idx], self.n_total[
                idx] = _corrcoeff_update(
                preds[:, idx], target[:,
                                      idx], self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx],
                self.corr_xy[idx], self.n_total[idx]
            )

    def compute(self):
        """Computes pearson correlation coefficient over state."""

        if self.mean_x[0].numel() > 1:  # multiple devices, need further reduction
            var_x = [0] * self.num_classes
            var_y = [0] * self.num_classes
            mean_x = [0] * self.num_classes
            mean_y = [0] * self.num_classes
            corr_xy = [0] * self.num_classes
            n_total = [0] * self.num_classes

            for idx in range(self.num_classes):
                var_x[idx], var_y[idx], corr_xy[idx], n_total[idx] = _final_aggregation(
                    self.mean_x[idx], self.mean_y[idx], self.var_x[idx], self.var_y[idx], self.corr_xy[idx],
                    self.n_total[idx]
                )
                mean_x[idx] = torch.mean(self.mean_x[idx])
                mean_y[idx] = torch.mean(self.mean_y[idx])
        else:
            var_x = self.var_x
            var_y = self.var_y
            mean_x = self.mean_x
            mean_y = self.mean_y
            corr_xy = self.corr_xy
            n_total = self.n_total

        ccc = [0] * self.num_classes
        for idx in range(self.num_classes):
            ccc[idx] = 2 * corr_xy[idx] / \
                (var_x[idx] + var_y[idx] + n_total[idx]
                 * (mean_x[idx] - mean_y[idx]).square())

        return torch.mean(torch.stack(ccc))
