"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

from pytorch_lightning.callbacks import BaseFinetuning
import pytorch_lightning as pl

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


def multiplicative(epoch):
    return 2


class MultiStageABAW3(BaseFinetuning):
    def __init__(self, unfreeze_temporal_at_epoch: int = 10,
                 lambda_func: Callable = multiplicative,
                 temporal_initial_ratio_lr: float = 10e-2,
                 temporal_initial_lr: Optional[float] = None,
                 should_align: bool = True,
                 initial_denom_lr: float = 10.0,
                 train_bn: bool = True,
                 verbose: bool = False,
                 rounding: int = 12, ):
        super(MultiStageABAW3, self).__init__()

        self.unfreeze_temporal_at_epoch: int = unfreeze_temporal_at_epoch
        self.lambda_func: Callable = lambda_func
        self.temporal_initial_ratio_lr: float = temporal_initial_ratio_lr
        self.temporal_initial_lr: Optional[float] = temporal_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_temporal_lr: Optional[float] = None

    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_temporal_lr": self.previous_temporal_lr,
        }

    def on_load_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            callback_state: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        self.previous_temporal_lr = callback_state["previous_temporal_lr"]
        super().on_load_checkpoint(trainer, pl_module,
                                   callback_state["internal_optimizer_metadata"])

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `temporal` attribute.
        """
        if hasattr(pl_module, "temporal") and isinstance(pl_module.temporal, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException(
            "The LightningModule should have a nn.Module `temporal` attribute")

    def finetune_function(
            self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        # TODO
        """Called when the epoch begins."""

        if epoch == self.unfreeze_temporal_at_epoch:
            # Un-freeze temporal module
            current_lr = optimizer.param_groups[0]["lr"]
            initial_temporal_lr = (
                self.temporal_initial_lr
                if self.temporal_initial_lr is not None
                else current_lr * self.temporal_initial_ratio_lr
            )
            self.previous_temporal_lr = initial_temporal_lr
            self.unfreeze_and_add_param_group(
                pl_module.temporal,
                optimizer,
                initial_temporal_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"temporal lr: {round(initial_temporal_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_temporal_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_temporal_lr = self.lambda_func(
                epoch + 1) * self.previous_temporal_lr
            next_current_temporal_lr = (
                current_lr
                if (self.should_align and next_current_temporal_lr > current_lr)
                else next_current_temporal_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_temporal_lr
            self.previous_temporal_lr = next_current_temporal_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"temporal lr: {round(next_current_temporal_lr, self.rounding)}")

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        # TODO:
        self.freeze(pl_module.temporal)
        # Do as usual
        pass
