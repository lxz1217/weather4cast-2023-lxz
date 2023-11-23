# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


from turtle import pos
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.evaluate import *        


class BaseLitModule(pl.LightningModule):
    def __init__(self, UNet_params, config):
        super(BaseLitModule, self).__init__()

        self.save_hyperparameters()
        self.config = config

        pos_weight = torch.tensor(config['pos_weight']);

        self.loss = config['loss']
        self.bs = config['batch_size']
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 'L1': nn.L1Loss(), 'mse': F.mse_loss,
            'BCELoss': nn.BCELoss()
        }[self.loss]
        self.main_metric = {
            'smoothL1':          'Smooth L1',
            'L1':                'L1',
            'mse':               'MSE',  # mse [log(y+1)-yhay]'
            'BCELoss':           'BCE',  # binary cross-entropy
            'BCEWithLogitsLoss': 'BCE with logits',
            'CrossEntropy':      'cross-entropy',
            }[self.loss]

        self.relu = nn.ReLU() # None
        t = f"============== n_workers: {config['n_workers']} | batch_size: {config['batch_size']} \n"+\
            f"============== loss: {self.loss} | weight: {pos_weight} (if using BCEwLL)"
        print(t)
    
    def on_fit_start(self):
        """ create a placeholder to save the results of the metric per variable """
        metric_placeholder = {self.main_metric: -1}
        self.logger.log_hyperparams(self.hparams, metric_placeholder)
        
    def forward(self, x):
        x = self.model(x)

        # if self.loss =='BCELoss':
        #     x = self.relu(x)
        return x

    def retrieve_only_valid_pixels(self, x, m):
        """ we asume 1s in mask are invalid pixels """
        ##print(f"x: {x.shape} | mask: {m.shape}")
        return x[~m]

    def get_target_mask(self, metadata):
        mask = metadata['target']['mask']
        #print("mask---->", mask.shape)
        return mask
    
    def _compute_loss(self, y_hat, y, agg=True, mask=None, reduction='mean'):
        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0
        # print("================================================================================")
        # print(y_hat.shape, y_hat.min(), y_hat.max())
        # print(y.shape, y.min(), y.max())
        if agg:
            loss = self.loss_fn(y_hat, y, reduction=reduction)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y, metadata  = batch
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)
        self.log('train_loss', loss)
        return loss
                
    def validation_step(self, batch, batch_idx, phase='val'):
        x, y, metadata  = batch
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)

        if mask is not None:
            y_hat[mask] = 0
            y[mask] = 0
    
        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values  = {'val_mse': loss} 
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
    
        return {'loss': loss.cpu(), 'N': x.shape[0],
                'mse': loss.cpu()}
    
    def validation_epoch_end(self, outputs, phase='val'):
        print("Validation epoch end average over batches: ",
              [batch['N'] for batch in outputs]);
        avg_loss = np.average([batch['loss'] for batch in outputs],
                              weights=[batch['N'] for batch in outputs]);
        avg_mse  = np.average([batch['mse'] for batch in outputs],
                              weights=[batch['N'] for batch in outputs]);
        values={f"{phase}_loss_epoch": avg_loss,
                f"{phase}_mse_epoch":  avg_mse}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        self.log(self.main_metric, avg_loss, batch_size=self.bs, sync_dist=True)

    def test_step(self, batch, batch_idx, phase='test'):
        x, y, metadata = batch
        y_hat = self.forward(x)
        mask = self.get_target_mask(metadata)
        loss = self._compute_loss(y_hat, y, mask=mask)
        
        if mask is not None:
            y_hat[mask]=0
            y[mask]=0

        #LOGGING
        self.log(f'{phase}_loss', loss, batch_size=self.bs, sync_dist=True)
        values = {'test_mse': loss}
        self.log_dict(values, batch_size=self.bs, sync_dist=True)
        
        return 0, y_hat

    def predict_step(self, batch, batch_idx, phase='predict'):
        x, y, metadata = batch
        y_hat = self(x)
        mask = self.get_target_mask(metadata)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.config["lr"]), weight_decay=float(self.config["weight_decay"])) 
        # optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config["train"]["lr"]))
        return optimizer

