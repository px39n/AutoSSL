import pytorch_lightning as pl
import torch
from torch import nn
from lightly.loss.vicreg_loss import VICRegLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules import SimSiamPredictionHead 
from autoSSL.models.Backbone import pipe_backbone
from autoSSL.models.get_loss import get_loss
from torch.utils.data import DataLoader
from lightly.data.dataset import LightlyDataset
from functools import partial    
from typing import List, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
#from torch.utils.data import DataLoader
from autoSSL.utils.knn import knn_predict
from autoSSL.utils.dim2head import dim2head
from torch.optim.lr_scheduler import LambdaLR
from lightly.models.modules import heads


class Toymodel(pl.LightningModule):
    def __init__(self, backbone="resnet18", 
                 stop_gradient: bool=False,
                 prjhead_dim: list=[2048, 2048],
                 predhead_dim: list=[],
                 loss_func: str ="NegativeCosineSimilarity",
                 views: int =2,
                 view_model: str="None",
                 MonitoringbyKNN=None,
                 knn_k: int = 200,
                 knn_t: float = 0.1,
                 optimizer: str="SGD",    
                 schedule: str="cos", 
                 batch = 256,
                 max_epochs= 100,
                 samples=50000,
 
                ):
        super().__init__()
        self.backbone, self.out_dim = pipe_backbone(backbone)
        self.prjhead_dim = prjhead_dim
        self.predhead_dim = predhead_dim
        self.loss_func = loss_func
        self.views = views
        self.view_model = view_model
        self.criterion = get_loss(loss_func)
        self.stop_gradient = stop_gradient 
        self.optimizer = optimizer
        self.schedule = schedule 
        self.bats= batch 
        self.max_epochs = max_epochs
        self.samples=samples
        self.debug={"stop":None, "optim":None}
        if self.prjhead_dim:
            self.projection_head = dim2head(self.prjhead_dim)
            
        if self.predhead_dim:
            self.prediction_head = dim2head(self.predhead_dim) 
        
        if MonitoringbyKNN:
            self.dataloader_kNN = MonitoringbyKNN[0]
            self.num_classes = MonitoringbyKNN[1]
            self.knn_k = knn_k
            self.knn_t = knn_t
            self.max_accuracy = 0.0
            self._train_features: Optional[Tensor] = None
            self._train_targets: Optional[Tensor] = None
            self._val_predicted_labels: List[Tensor] = []
            self._val_targets: List[Tensor] = []
        else:
            self.dataloader_kNN=None
            
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        if self.prjhead_dim:
            x = self.projection_head(x)
            
        return x

    def forward_with_p(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        p = f
        
        if self.prjhead_dim:
            f = self.projection_head(f)
            p = self.projection_head(p)
    
        if self.predhead_dim:
            p = self.prediction_head(p)
            
        if self.stop_gradient:
            f = f.detach()
        return f, p
    
    def training_step(self, batch, batch_idx):
        
        if self.view_model=="None":
            
            (x0, x1), _, _ = batch
            if self.stop_gradient or self.predhead_dim:
                z0, p0 = self.forward_with_p(x0)
                z1, p1 = self.forward_with_p(x1)
                loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
                self.debug["stop"]=1
            else:
                z0 = self.forward(x0)
                z1 = self.forward(x1)
                loss = self.criterion(z0, z1)
                    # Compute the std of z0 and z1 and log it
                self.debug["stop"]=2
            self.log('representation_std', variance_loss(z0))
            self.log('correlation', covariance_loss(z0))
            
           
        else:
            views, _, _ = batch
            features = [self.forward_with_p(view) for view in views]   # features= [...,batch, feature for view_i]

            zs = torch.stack([z for z, _ in features])     # zs= [...,embedding of batch, feature for view_i]
            ps = torch.stack([p for _, p in features])     # zs= [...,projection of batch, feature for view_i]
            
            loss = 0.0
            self.debug["stop"]=3
            if self.view_model=="fastsim":     # fastsim #pair-pair #1_n # mean_n
                self.debug["stop"]=4
                for i in range(len(views)):
                    mask = torch.arange(len(views), device=self.device) != i
                    mean_embed= torch.mean(zs[mask], dim=0)
                    loss += self.criterion(ps[i],mean_embed) / len(views)     # ps[i]= embedding of batch, feature for view_i

            elif self.view_model=="pair-pair":  # fastsim #pair-pair #1_n # mean_n
                self.debug["stop"]=5
                for i in range(len(views)):
                    for j in range(len(views)):
                        if i!=j:
                            loss += self.criterion(ps[i], zs[j]) / (len(views)*len(views)/2)
                            
            elif self.view_model=="1_n":     # fastsim #pair-pair #1_n # mean_n #1_fastsim
                self.debug["stop"]=6
                for i in range(1,len(views)):
                    loss += self.criterion(ps[0], zs[i]) / (len(views)-1)  
                    
            elif self.view_model=="1_fastsim":     # fastsim #pair-pair #1_n # mean_n #1_fastsim
                self.debug["stop"]=7
                mask = torch.arange(len(views), device=self.device) != 0
                loss = self.criterion(ps[0], torch.mean(zs[mask], dim=0)) 
                
            elif self.view_model=="mean_n":     # fastsim #pair-pair #1_n # mean_n #1_fastsim
                self.debug["stop"]=8
                mean_embed=torch.mean(zs, dim=0)
                for i in range(1,len(views)):
                    loss += self.criterion(ps[i], mean_embed) / (len(views))  
            else:
                ValueError(f"Unknown model name: {self.view_model}")
            #self.log('representation_std', features[0].std())
            self.log('representation_std', variance_loss(zs[0]))
            self.log('correlation', covariance_loss(zs[0]))
            
        self.log('train_loss', loss)
        
        return loss

    
    
    def configure_optimizers(self):
        
        samples=self.samples  
        bats=self.bats 
        max_epochs=self.max_epochs 
         
        self.debug["optim"]=[samples,bats,max_epochs ]
        if self.optimizer=="LARS":

            optim  = LARS(
            self.parameters(),
            lr=0.2 * bats / 128,  # Initialize with a LR of 0
            weight_decay=1.5 * 1e-6,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm
        )
            
        elif self.optimizer=="Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=6e-2 )
        elif self.optimizer=="SGD":
            optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2 * bats / 128,
            momentum=0.9,
            weight_decay=5e-4,
        )
        
        if self.schedule=="cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        elif self.schedule=="LambdaLR":
            warmup_steps = samples/bats * 10 
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optim,
                    linear_warmup_decay(warmup_steps),
                ),
                "interval": "step",
                "frequency": 1,
            }
        
        return [optim], [scheduler]
    
    
  
    def on_validation_epoch_start(self) -> None:
        if self.dataloader_kNN:
            train_features = []
            train_targets = []
            with torch.no_grad():
                for data in self.dataloader_kNN:
                    img, target, _ = data
                    img = img.to(self.device)
                    target = target.to(self.device)
                    feature = self.backbone(img).squeeze()
                    feature = F.normalize(feature, dim=1)
                    if (
                        dist.is_available()
                        and dist.is_initialized()
                        and dist.get_world_size() > 0
                    ):
                        # gather features and targets from all processes
                        feature = torch.cat(dist.gather(feature), 0)
                        target = torch.cat(dist.gather(target), 0)
                    train_features.append(feature)
                    train_targets.append(target)
            self._train_features = torch.cat(train_features, dim=0).t().contiguous()
            self._train_targets = torch.cat(train_targets, dim=0).t().contiguous()

    def validation_step(self, batch, batch_idx) -> None:
        if self.dataloader_kNN:
            # we can only do kNN predictions once we have a feature bank
            if self._train_features is not None and self._train_targets is not None:
                images, targets, _ = batch
                feature = self.backbone(images).squeeze()
                feature = F.normalize(feature, dim=1)
                predicted_labels = knn_predict(
                    feature,
                    self._train_features,
                    self._train_targets,
                    self.num_classes,
                    self.knn_k,
                    self.knn_t,
                )
                if dist.is_initialized() and dist.get_world_size() > 0:
                    # gather predictions and targets from all processes
                    predicted_labels = torch.cat(dist.gather(predicted_labels), 0)
                    targets = torch.cat(dist.gather(targets), 0)

                self._val_predicted_labels.append(predicted_labels.cpu())
                self._val_targets.append(targets.cpu())

    def on_validation_epoch_end(self) -> None:
        if self.dataloader_kNN:
            if self._val_predicted_labels and self._val_targets:
                predicted_labels = torch.cat(self._val_predicted_labels, dim=0)
                targets = torch.cat(self._val_targets, dim=0)
                top1 = (predicted_labels[:, 0] == targets).float().sum()
                acc = top1 / len(targets)
                if acc > self.max_accuracy:
                    self.max_accuracy = acc.item()
                self.log("kNN_accuracy", acc * 100.0, prog_bar=True)

            self._val_predicted_labels.clear()
            self._val_targets.clear()
            
            
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)



import math
import torch


class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
        return loss
    
def exclude_bias_and_norm(p):
    return p.ndim == 1


def covariance_loss(x: Tensor) -> Tensor:
    """Returns VICReg covariance loss.

    Generalized version of the covariance loss with support for tensors with more than
    two dimensions. Adapted from VICRegL:
    https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L299

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
    """
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    # cov has shape (..., dim, dim)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()
                     
                     
import torch.nn.functional as F
def variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns VICReg variance loss.

    Args:
        x:
            Tensor with shape (batch_size, ..., dim).
        eps:
            Epsilon for numerical stability.
    """
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(std)
    return loss

