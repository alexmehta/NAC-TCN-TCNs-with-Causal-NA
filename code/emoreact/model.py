"""Alexander Mehta"""
from typing import Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from core.nt_tcn import NACTempConv as TemporalConvNet
# from core.tcn import TemporalConvNet
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, ROC, AUROC
from torchmetrics.classification import MulticlassF1Score
from PIL import Image
from torchmetrics.classification import BinaryF1Score
from torch.optim import lr_scheduler
from core.lstm import GRU, LSTM
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
from torch.nn import functional as F
import pandas as pd
from core import vggface2
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics import ConcordanceCorrCoef
import matplotlib.pyplot as plt
# import required libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import uuid
import random
import math
import time
import datetime
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from core.atcn import ATempConv 

class TCNModel(LightningModule):

    def get_backbone(self):
        backbone_name = 'regnet_y-800mf'
        regnet_backbone_dict = {'400mf': (regnet_y_400mf, 440), '800mf': (regnet_y_800mf, 784),
                                '1.6gf': (regnet_y_1_6gf, 888), '3.2gf': (regnet_y_3_2gf, 1512)}

        bb_model = regnet_backbone_dict[backbone_name.split(
            '-')[-1]][0](pretrained=True)
        backbone = create_feature_extractor(
            bb_model, return_nodes={'flatten': 'feat'})
        # regnet_y: 400mf = 440, 800mf = 784, regnet_y_1_6gf = 888
        # regnet_x: 400mf = 400
        num_feats = {
            'feat': regnet_backbone_dict[backbone_name.split('-')[-1]][1]}

        # backbone_freeze = ['block2','block3']
        backbone_freeze = []
        if len(backbone_freeze) > 0:
            # Freeze backbone model
            for named, param in backbone.named_parameters():
                do_freeze = True
                if 'all' not in backbone_freeze or not (
                        isinstance(param, nn.BatchNorm2d)):
                    for layer_name in backbone_freeze:
                        if layer_name in named:
                            do_freeze = True
                            break
                if do_freeze:
                    param.requires_grad = False
        num_feats = {'feat': num_feats['feat']}
        return backbone, num_feats

    def __init__(self):
        super().__init__()
        # self.temporal = TemporalConvNet(
            # 784, [440]*1, kernel_size=7, dropout=0.5, dilation_size=4)
        # self.temporal = TemporalConvNet(
            #  784, [128]*3, kernel_size=11, dropout=0.5, dilation_size=4)
        self.temporal = nn.MultiheadAttention(784, 16, dropout=0.5)
        # self.temporal = ATempConv(
            #  784, [128]*3, kernel_size=9, dropout=0.5, dilation_size=4)
        #self.temporal =  GRU(784, 440, 1, dropout=0.5)
        # self.temporal = LSTM(784, 440, 1, dropout=0.5)
        self.fc = nn.Linear(784, 8)
        self.loss_a = nn.MSELoss()
        self.loss_b = nn.BCEWithLogitsLoss()
        self.correct = 0
        self.total = 0
        self.relu = nn.ReLU()
        self.seq_len = 128
        self.drop = nn.Dropout(0.5)
        self.accuracy = Accuracy()
        self.auroc = AUROC(num_classes=8, average='macro')
        # self.ccc = ConcordaneeCorrCoef()
        self.f1 = F1Score()
        self.feat = self.get_backbone()[0]
    def saliency_map(self, input_seq):
        x = self(input_seq)
        out,_ = torch.max(x, dim=1)
        loss = self.loss_a(out,torch.ones_like(out))
        loss.backward()
        
        print(x.grad,out.grad,input_seq.grad)
        saliency = torch.amax(input_seq.grad.data.abs(), (2, 3,4))
        return saliency

    def visualize(self, input_seq, image_path_seq):
        input_seq = input_seq.unsqueeze(0)
        temporal_salency_map = self.saliency_map(input_seq).unsqueeze(0)
        temporal_salency_map = temporal_salency_map.detach().cpu().numpy()
        temporal_salency_map = temporal_salency_map/temporal_salency_map.max()
        temporal_salency_map = temporal_salency_map.squeeze(0)

        # return line plot with images attached
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(temporal_salency_map)
        ax.set_xlabel('Time')
        ax.set_ylabel('Saliency')
        ax.set_title('Temporal Saliency Map')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 128)
        plt.tick_params(axis='both', which='major', labelsize=10)
        ax.grid()
        plt.figure(dpi=1200)
        # place images on top of plot
        for i in range(len(image_path_seq)-1):
            if(image_path_seq[i] == '' or None):
                continue
            try:
                img = Image.open(image_path_seq[i])
            except:
                continue
            img = img.resize((224, 224))
            img = np.array(img)
            img = img/255
            imagebox = OffsetImage(img, zoom=0.1)
            imagebox.image.axes = ax
            print(i, temporal_salency_map[0][i], image_path_seq[i])
            ab = AnnotationBbox(imagebox, (i, temporal_salency_map[0][i]*100),
                                xybox=(0., -16.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.4,
                                arrowprops=dict(
                                arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                )
            ax.add_artist(ab)

        # save figure to folder with a unique id

        folder = 'saliency_map'
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, str(uuid.uuid4())+'.png')
        fig.savefig(path)

        # return the path to the folder
        del fig
        del ax
        del temporal_salency_map
        torch.cuda.empty_cache()
        return path

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # Convert to batch size * seq x 3 x h x w for feature extraction
        num_seq = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3)
        feat = torch.reshape(
            x, (num_seq * self.seq_len,) + x.shape[2:])
        feat = self.feat(feat)['feat']
        feat = torch.reshape(feat, (num_seq, self.seq_len, -1))
        # feat = feat.permute(0, 2, 1).contiguous()
        x,_ = self.temporal(feat,feat,feat)
        x = x[:,-1,:]
        x = self.relu(x)
        # print(x.shape)
        x = self.drop(x)

        x = self.fc(x)
        return x

    def _shared_eval(self, batch, batch_idx, cal_loss=False):

        targets = torch.Tensor(batch['labels'])

        inputs = batch['feat'].float()
        # print(inputs.shape)
        out = self(inputs)
        targs = targets
        loss = 0
        # loss += self.loss_a(out[:,8], targs[:,8])
        loss += self.loss_b(out[:, 0:8], targs[:, 0:8])
        return out, loss/2

    def training_step(self, batch, batch_idx):
        # print(self.visualize(batch['feat'][0], batch['image_list'][0]))
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        self.log('train_loss', loss)
        return loss

    def test_epoch_end(self, outputs):
        # self.log('ccc', self.ccc.compute())
        self.log('f1', self.f1.compute())
        self.log('accuracy', self.accuracy.compute())
        self.log('auroc', self.auroc.compute())
        # iterate through self.output and self.target and save to csv order pair
        # calculate mse with outputs
        #
        #
        mse = 0
        for i in range(len(outputs)):
            mse += (outputs[i, 0]-outputs[i, 1])**2
        mse = mse/len(outputs)
        self.log('mse', mse)
        return {'f1': self.f1.compute(), 'accuracy': self.accuracy.compute(), 'auroc': self.auroc.compute(), 'mse': mse}

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=True)
        self.accuracy.update((F.sigmoid(out[:, 0:8]).unsqueeze(1) > 0.5).float(
        ).flatten(), batch['labels'].float()[:, 0:8].flatten().int())
        self.auroc.update(out[:, 0:8], batch['labels'][:, 0:8].int())

    def validation_epoch_end(self, outputs):
        self.log('accuracy', self.accuracy.compute())
        self.log('auroc', self.auroc.compute())
        d = {'val_accuracy': self.accuracy.compute(
        ), 'val_auroc': self.auroc.compute()}
        self.accuracy.reset()
        self.auroc.reset()
        print(d)
        return d

    def test_step(self, batch, batch_idx):
        # f1 score of multi binary classification
        targets = batch['labels'].float()
        inputs = batch['feat'].float()
        out = self(inputs)
        out = out.squeeze(1)
        self.accuracy.update((F.sigmoid(out[:, 0:8]).unsqueeze(
            1) > 0.5).float().flatten(), targets[:, 0:8].flatten().int())
        self.f1.update((F.sigmoid(out[:, 0:8]).unsqueeze(
            1) > 0.5).float().flatten(), targets[:, 0:8].flatten().int())
        self.auroc.update(out[:, 0:8], targets[:, 0:8].int())
        # choose random from batch to create saliency map

        return targets, inputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        opt_lr_dict = {'optimizer': optimizer}
        min_lr = 0.0001 * 0.1
        t_o = 20 * 2
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=t_o, T_mult=2, eta_min=min_lr, last_epoch=-1)
        opt_lr_dict.update(
            {'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})
        return opt_lr_dict

    def train_dataloader(self):
        return super().train_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
    # 1d temporal saliency map

 