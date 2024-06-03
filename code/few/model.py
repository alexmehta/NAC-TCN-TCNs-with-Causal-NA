"""Alexander Mehta"""
from nt_tcn import TemporalConvNet
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, F1Score, ROC, AUROC 
from torch.optim import lr_scheduler
from lstm import GRU, LSTM
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf
import random
from torch.nn import functional as F
import pandas as pd
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics import ConcordanceCorrCoef, PearsonCorrCoef

class TCNModel(LightningModule):

    def __init__(self):
        super().__init__()
        # self.temporal = TemporalConvNet(440,[32]*3,kernel_size=11,dropout=0.2,dilation_size=2)
        self.temporal =  GRU(440, 128, 2, dropout=0.5)
        # self.temporal = LSTM(440, 128, 2, dropout=0.5)
        self.fc = nn.Linear(440, 20*2)
        self.loss_a = nn.CrossEntropyLoss()
        self.loss_b = nn.CrossEntropyLoss()
        self.loss_c = nn.MSELoss()
        self.correct = 0
        self.total = 0
        self.relu = nn.ReLU()
        self.seq_len =32 
        self.drop = nn.Dropout(0.5)
        self.accuracy_valence = PearsonCorrCoef()
        self.accuracy_arousal = PearsonCorrCoef()


    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # Convert to batch size * seq x 3 x h x w for feature extraction
        num_seq = x.shape[0]
        x = x.permute(0,1,4,2,3)
        feat = torch.reshape(
        x, (num_seq * self.seq_len,) + x.shape[2:])
        feat = self.feat(feat)['feat']
        feat = torch.reshape(feat,(num_seq,self.seq_len,-1))
        feat = feat.permute(0,2,1).contiguous()
        feat =  self.drop(feat)
        # print(feat.shape)
        x = self.temporal(feat)
        # print(x.shape)
        #x = x.permute(0,2,1).contiguous()
        # x = self.relu(x).permute(0,2,1).contiguous()
        x = self.relu(x)
        x = self.fc(x)
        # print(x.shape)
        return x
    def _shared_eval(self, batch, batch_idx, cal_loss=False):
        targets_a = torch.Tensor(batch['valence_label'])
        targets_b = torch.Tensor(batch['arousal_label'])
        inputs = batch['clip'].float()
        out = self(inputs)
        loss =0
        # reshape targets_a, targets_b to (batch_size * 32) and out to (batch_size,6)
        a,b,c = out.shape
        out = out.view(a*b,c)
        a,b, = targets_a.shape
        targets_a = targets_a.view(a*b)
        targets_b = targets_b.view(a*b)
        # print(out.shape,targets_a.shape,targets_b.shape)
        # print(targets_a.max(),targets_a.min(),targets_b.max(),targets_b.min())
        import random
        
        loss += self.loss_a(out[:,0:20],targets_a)
        loss += self.loss_b(out[:,20:], targets_b)
        # loss += self.loss_c(out[:,0:3],targets_a)
        # loss += self.loss_c(out[:,3:], targets_b)
        return out, loss/4.0
    def training_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch,batch_idx,cal_loss=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_epoch_end(self, outputs):
        self.log('accuracy_valence', self.accuracy_valence.compute())
        self.log('accuracy_arousal', self.accuracy_arousal.compute())

        # save avg to file 

        avg = (self.accuracy_valence.compute()+self.accuracy_arousal.compute())/2
        self.log('avg_test_acc',avg)

        with open('avg_test_acc_tcn.txt','a') as f:
            write = (self.accuracy_arousal.compute().item(),self.accuracy_valence.compute().item(),avg.item())
            # tuple to string 
            write = str(write).replace('(','').replace(')','').replace(',','')
            f.write(write + "\n")
        return {'accuracy_valence': self.accuracy_valence.compute(),'accuracy_arousal': self.accuracy_arousal.compute(),'avg_test_acc':(self.accuracy_valence.compute()+self.accuracy_arousal.compute())/2}
    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch,batch_idx,cal_loss=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        # get value based on class
        out_a, out_b = out[:,0:20], out[:,20:]
        out_a = torch.argmax(out_a,dim=1).float()
        out_b = torch.argmax(out_b,dim=1).float()
        self.accuracy_valence.update(out_a/10,batch['valence_label'].view(-1).float()/10)
        self.accuracy_arousal.update(out_b/10,batch['arousal_label'].view(-1).float()/10)
    def validation_epoch_end(self, outputs):
        self.log('accuracy_valence_val', self.accuracy_valence.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('accuracy_arousal_val', self.accuracy_arousal.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val',(self.accuracy_valence.compute()+self.accuracy_arousal.compute())/2.0,logger=True,prog_bar=True)
        # iterate through self.output and self.target and save to csv order pair
        d = {'accuracy_valence_val': self.accuracy_valence.compute(),'accuracy_arousal_val': self.accuracy_arousal.compute(),'avg_val':(self.accuracy_valence.compute()+self.accuracy_arousal.compute())/2.0}
        self.accuracy_valence.reset()
        self.accuracy_arousal.reset()
        return  d
    def test_step(self,batch,batch_idx):
        out, loss = self._shared_eval(batch,batch_idx,cal_loss=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        out_a, out_b = out[:,0:20], out[:,20:]
        out_a = torch.argmax(out_a,dim=1).float()
        out_b = torch.argmax(out_b,dim=1).float()
        self.accuracy_valence.update(out_a/10,batch['valence_label'].view(-1).float()/10)
        self.accuracy_arousal.update(out_b/10,batch['arousal_label'].view(-1).float()/10)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        opt_lr_dict = {'optimizer': optimizer}
        min_lr = 0.0001 * 0.1
        t_o = 34* 2
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_o, T_mult=2, eta_min=min_lr, last_epoch=-1)
        opt_lr_dict.update(
            {'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})
        return opt_lr_dict
    def train_dataloader(self):
        return super().train_dataloader()
    def test_dataloader(self):
        return super().test_dataloader()
