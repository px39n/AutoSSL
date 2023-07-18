import pytorch_lightning as pl
import torch
from torch import nn
from autoSSL.models.Backbone import pipe_backbone
from autoSSL.models.get_loss import get_loss, Magic_Cube
from torch.utils.data import DataLoader
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
import random
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
import copy
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
                 momentum: bool= False,
                 learn_rate:float = None,
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
        self.momentum=momentum
        self.learn_rate=learn_rate
        
        if self.learn_rate:
            print("Initialize with Customized Learning rate"+str(self.learn_rate))
            
        
        if self.prjhead_dim:
            self.projection_head = dim2head(self.prjhead_dim)
            
        if self.predhead_dim:
            self.prediction_head = dim2head(self.predhead_dim) 
        
        if self.momentum:
            self.backbone_momentum = copy.deepcopy(self.backbone)
            deactivate_requires_grad(self.backbone_momentum)
            if self.prjhead_dim:
                self.projection_head_momentum = copy.deepcopy(self.projection_head)
                deactivate_requires_grad(self.projection_head_momentum)
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
            self.p_test=MonitoringbyKNN[2]
        else:
            self.dataloader_kNN=None
            self.p_test=None
            
    def forward(self, x):
        a = self.backbone(x).flatten(start_dim=1)
        
        if self.prjhead_dim:
            f = self.projection_head(a)
        else:
            f=a
            
        if self.momentum: 
            m=self.backbone_momentum(x).flatten(start_dim=1)
            if self.prjhead_dim:
                m=self.projection_head_momentum(m)
            
        if self.stop_gradient or self.predhead_dim or self.momentum:
            p=f
            if self.momentum:
                f=m
            if self.predhead_dim:
                p = self.prediction_head(p)
            if self.stop_gradient:
                f = f.detach()
            return f,p,a
        else:
            return f,f,a

    def forward_with_p(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        p = f
        
        if self.prjhead_dim:
            f = self.projection_head(f)
            p = f

        if self.predhead_dim:
            p = self.prediction_head(p)
            
        if self.stop_gradient:
            f = f.detach()
            
        return f, p
    
    def training_step(self, batch, batch_idx):
        
        if self.momentum:
                #momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
                update_momentum(self.backbone, self.backbone_momentum, m=0.99)
                if self.prjhead_dim:
                    update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)   
 
        views, _, _ = batch
        if self.view_model=="None":

            (x0, x1)=views[0:2]
             
            z0, p0,_ = self.forward(x0)
            z1, p1,_ = self.forward(x1)
            if self.stop_gradient or self.predhead_dim or self.momentum :
                loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
                self.debug["stop"]=1

            else: 
                loss= self.criterion(z0, z1)
                #loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
                self.debug["stop"]=2   
            #self.log('representation_std', variance_loss(z0))
            #self.log('correlation', covariance_loss(z0))
            #self.log('view_variance', view_variance(torch.stack([z for z in [z0,z1]])))
            features=[]
           
        else:
            
            features = [self.forward(view) for view in views]   # features= [...,batch, feature for view_i]
            zs = torch.stack([z for z, _,_ in features])     # zs= [...,embedding of batch, feature for view_i]
            ps = torch.stack([p for _, p,_ in features])     # zs= [...,projection of batch, feature for view_i]
            
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
                            
            elif self.view_model=="1_n":     # fastsim #pair-pair #1_n # mean_n #1_mean
                self.debug["stop"]=6
                for i in range(1,len(views)):
                    loss += self.criterion(ps[0], zs[i]) / (len(views)-1)  
                    
            elif self.view_model=="1_mean":     # fastsim #pair-pair #1_n # mean_n #1_mean
                self.debug["stop"]=7
                mean_embed=torch.mean(zs, dim=0)
                loss = self.criterion(ps[0], mean_embed ) 
                
            elif self.view_model=="mean_n":     # fastsim #pair-pair #1_n # mean_n #1_mean
                self.debug["stop"]=8
                mean_embed=torch.mean(zs, dim=0)
                for i in range(len(views)):
                    loss += self.criterion(ps[i], mean_embed) / (len(views))    
                    
            elif self.view_model=="std_view":     # fastsim #pair-pair #1_n # mean_n #1_mean
                variance=0
                self.debug["stop"]=9
                mean=torch.mean(zs, dim=0)
                #dev=0
                #if self.loss_func =="NegativeCosineSimilarity":
                #    dev=1
                for i in range(len(views)):
                    distance= self.criterion(ps[i], mean)**2
                    variance += distance / (len(views))  
                loss=torch.sqrt(variance) 
                        
            elif self.view_model=="var_view":     # fastsim #pair-pair #1_n # mean_n #1_mean #
                variance=0
                self.debug["stop"]=10
                mean=torch.mean(zs, dim=0)
                #dev=0
                #if self.loss_func =="NegativeCosineSimilarity":
                #    dev=1
                for i in range(len(views)):
                    distance=torch.exp(2* self.criterion(ps[i], mean))
                    variance += distance / (len(views))  
                loss=variance
                
            elif self.view_model=="1_mean":     # fastsim #pair-pair #1_n # mean_n #1_mean
                self.debug["stop"]=11
                mean_embed=torch.mean(zs, dim=0)
                loss = self.criterion(ps[0], mean_embed ) 
                
            elif self.view_model=="n_mean":     # fastsim #pair-pair #1_n # mean_n #1_mean 
 
                self.debug["stop"]=12
                mean_embed=torch.mean(ps, dim=0)
                loss =self.criterion(zs[0],mean_embed)
                
            elif self.view_model=="n_mean_sym":     # fastsim #pair-pair #1_n # mean_n #1_mean  n_mean n_mean_sym
                self.debug["stop"]=13
                mean_embed=torch.mean(zs, dim=0)
                mean_embed2=torch.mean(ps, dim=0)
                loss = (self.criterion(ps[0], mean_embed)+self.criterion( mean_embed2,zs[0]))/2
                
                                
                
                
                
            elif self.view_model=="me_me":     # fastsim #pair-pair #1_n # mean_n #1_mean
                self.debug["stop"]=12

                # Get the number of examples
                num_examples = zs.size(0)
                # If the number of examples is odd, round down to the nearest even number
                if num_examples % 2 != 0:
                    num_examples -= 1
                    
                # Compute the mean embedding for each half
                mean_embed_first = torch.mean(ps[:num_examples // 2], dim=0)
                mean_embed_last = torch.mean(zs[num_examples // 2: num_examples], dim=0)

                # Compute the loss
                loss = self.criterion(mean_embed_first, mean_embed_last)
                
                
            elif self.view_model=="test":      
                loss= self.criterion(ps) 
                    
            else:
                ValueError(f"Unknown model name: {self.view_model}")
            #self.log('representation_std', features[0].std())
            
        import time
        if not features:
            features = [self.forward(view) for view in views]   # features= [...,batch, feature for view_i]
            zs = torch.stack([z for z, _ , _ in features])     # zs= [...,embedding of batch, feature for view_i]
        xs = torch.stack([x for _, _ , x in features])     # zs= [...,embedding of batch, feature for view_i]
          
        start = time.time()
        

        self.log('std_batch', std_batch(zs))
        self.log('std_view', std_view(zs))
        self.log('std_feature', std_feature(zs))
        self.log('train_loss', loss)

        
        self.log('Cor_view_jk_10', Cor_view_jk_10(zs[:,0:2,:]))
        self.log('Cor_view_jk_01', Cor_view_jk_01(zs)) #All good
        self.log('Cor_view_jk_00', Cor_view_jk_00(zs)) #All good
        
        self.log('Cor_batch_ik_10', Cor_batch_ik_10(zs[0:2]))
        self.log('Cor_batch_ik_00', Cor_batch_ik_00(zs))
        self.log('Cor_batch_ik_01', Cor_batch_ik_01(zs))
 
        self.log('Cov_batch_ik_10', Cov_batch_ik_10(zs[0:2]))
        self.log('Cov_batch_ik_00', Cov_batch_ik_00(zs))
        self.log('Cov_batch_ik_01', Cov_batch_ik_01(zs))

        self.log('Cor_feature_ij_10', Cor_feature_ij_10(zs[0:2]))
        self.log('Cor_feature_ij_00', Cor_feature_ij_00(zs))
        self.log('Cor_feature_ij_01', Cor_feature_ij_01(zs))

        
        self.log('std_batch_emb', std_batch(xs))
        self.log('std_view_emb', std_view(xs))
        self.log('std_feature_emb', std_feature(zs))
 
        self.log('Cor_view_jk_10_emb', Cor_view_jk_10(xs[:,0:2,:]))
        self.log('Cor_view_jk_01_emb', Cor_view_jk_01(xs)) #All good
        self.log('Cor_view_jk_00_emb', Cor_view_jk_00(zs)) #All good
        
        self.log('Cor_batch_ik_10_emb', Cor_batch_ik_10(xs[0:2]))
        self.log('Cor_batch_ik_00_emb', Cor_batch_ik_00(xs))
        self.log('Cor_batch_ik_01_emb', Cor_batch_ik_01(xs))
 
        self.log('Cov_batch_ik_10_emb', Cov_batch_ik_10(xs[0:2]))
        self.log('Cov_batch_ik_00_emb', Cov_batch_ik_00(xs))
        self.log('Cov_batch_ik_01_emb', Cov_batch_ik_01(xs))

        self.log('Cor_feature_ij_10_emb', Cor_feature_ij_10(xs[0:2]))
        self.log('Cor_feature_ij_00_emb', Cor_feature_ij_00(xs))
        self.log('Cor_feature_ij_01_emb', Cor_feature_ij_01(xs))
         
        
        self.log('time_log', time.time() - start)
        
        #  Cor_view_jk_01  Cor_view_jk_00    Cor_view_jk_10 Cor_batch_ik_00 Cor_batch_ik_01 Cor_batch_ik_10 Cor_feature_ij_00 Cor_feature_ij_01 Cor_feature_ij_10
        
        
        
        
        return loss

    
    
    def configure_optimizers(self):
        
        samples=self.samples  
        bats=self.bats 
        max_epochs=self.max_epochs 
         

        
        if self.learn_rate:
            self.learn_rate=self.learn_rate
        elif self.optimizer=="LARS":
            self.learn_rate=0.2 * bats / 128
        elif self.optimizer=="Adam":
            self.learn_rate=6e-2
        elif self.optimizer=="SGD":
            self.learn_rate=6e-2 * bats / 128
        else:
            raise ValueError("Wrong Learning Rate")
        self.debug["optim"]=[samples,bats,max_epochs,self.learn_rate]
        if self.optimizer=="LARS":
            optim  = LARS(
            self.parameters(),
            lr=self.learn_rate,  # Initialize with a LR of 0
            weight_decay=1.5 * 1e-6,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm
        )
            
        elif self.optimizer=="Adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.learn_rate )
        elif self.optimizer=="SGD":
            optim = torch.optim.SGD(
            self.parameters(),
            lr=self.learn_rate,
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


from typing import List 
import torch.nn.functional as F
 
def Cov_batch_ik_10(x: torch.Tensor) -> torch.Tensor: # 
    res = 0
    
    for i in range(len(x)):
        view = x[i]

        dim = view.size(0)
        # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(dim, device=view.device, dtype=torch.bool)
        # cov has shape (..., dim, dim)
        cov= torch.cov(view)
        loss = cov[..., nondiag_mask].pow(2).mean() 
        res += loss/len(x)
    return res


def Cov_batch_ik_01(x: torch.Tensor) -> torch.Tensor:  
    x = x.permute(0, 2, 1)  
    X1=x[0]
    X2=x[1]
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True) 
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1)  
    X2_centered_normalized = (X2 - mean_X2)  
    dim=X1.size(0)
    nondiag_mask =  torch.eye(dim, device=x.device, dtype=torch.bool)    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].pow(2).mean()

    return loss  
 
def Cov_batch_ik_00(x: torch.Tensor) -> torch.Tensor: 
    x = x.permute(0, 2, 1) 
    X1=x[0]
    X2=x[1]
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True) 
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) 
    X2_centered_normalized = (X2 - mean_X2) 
    dim=X1.size(0)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].pow(2).mean()

    return loss 
 
def std_batch(x: torch.Tensor) -> torch.Tensor:
    """Calculate the standard deviation across the batch dimension and then average."""
    return x.std(dim=1).mean()

def std_view(x: torch.Tensor) -> torch.Tensor:
    """Calculate the standard deviation across the view dimension and then average."""
    return x.std(dim=0).mean()

def std_feature(x: torch.Tensor) -> torch.Tensor:
    """Calculate the standard deviation across the feature dimension and then average."""
    return x.std(dim=2).mean()



def Cor_feature_ij_10(x: torch.Tensor) -> torch.Tensor:
    #x = x.permute(0, 1, 2) 
    res = 0
    for i in range(len(x)):
        view = x[i]
        dim = view.size(0)
        # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(dim, device=view.device, dtype=torch.bool)
        # cov has shape (..., dim, dim)
        cov= torch.corrcoef(view)
        
        loss = cov[..., nondiag_mask].abs().nanmean()
        res += loss/len(x)
    return res



def Cor_feature_ij_01(x: torch.Tensor) -> torch.Tensor:  
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask =  torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 

def Cor_feature_ij_00(x: torch.Tensor) -> torch.Tensor:
    #x = x.permute(0, 1, 2) 
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 


def Cor_batch_ik_10(x: torch.Tensor) -> torch.Tensor: # 
    x = x.permute(0, 2, 1) 
    res = 0
    for i in range(len(x)):
        view = x[i]

        dim = view.size(0)
        # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(dim, device=view.device, dtype=torch.bool)
        # cov has shape (..., dim, dim)
        cov= torch.corrcoef(view)
        loss = cov[..., nondiag_mask].abs().nanmean() 
        res += loss/len(x)
    return res


 
def Cor_batch_ik_01(x: torch.Tensor) -> torch.Tensor:  
    x = x.permute(0, 2, 1)  
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask =  torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 

def Cor_batch_ik_00(x: torch.Tensor) -> torch.Tensor: 
    x = x.permute(0, 2, 1) 
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 


def Cor_view_jk_10(x: torch.Tensor) -> torch.Tensor:  
    
    x = x.permute(1, 2, 0) 
    res = 0
    for i in range(len(x)):
        view = x[i] 
        dim = view.size(0)
        # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(dim, device=view.device, dtype=torch.bool)
        # cov has shape (..., dim, dim)
        cov= torch.corrcoef(view)
        loss = cov[..., nondiag_mask].abs().nanmean() 
        res += loss/len(x)
    return res


def Cor_view_jk_01(x: torch.Tensor) -> torch.Tensor:  
    x = x.permute(1, 2, 0) 
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask =  torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 

def Cor_view_jk_00(x: torch.Tensor) -> torch.Tensor:
    x = x.permute(1, 2, 0) 
    X1=x[0]
    X2=x[1]
    
    # Compute the mean of X1 and X2 along the feature dimension
    mean_X1 = torch.mean(X1, dim=1, keepdim=True)
    mean_X2 = torch.mean(X2, dim=1, keepdim=True)
    
    # Compute the standard deviation of X1 and X2 along the feature dimension
    std_X1 = torch.std(X1, dim=1, keepdim=True)
    std_X2 = torch.std(X2, dim=1, keepdim=True)
    
    # Center and normalize the data by subtracting the mean and dividing by the standard deviation
    X1_centered_normalized = (X1 - mean_X1) / std_X1 
    X2_centered_normalized = (X2 - mean_X2) / std_X2 
    dim=X1.size(0)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)    
    
    # Compute the cross-correlation matrix
    cross_correlation = torch.matmul(X1_centered_normalized, X2_centered_normalized.t())/(X1.size(1)-1)
    
    loss = cross_correlation[..., nondiag_mask].abs().nanmean()

    return loss 
