from lightly.loss import NegativeCosineSimilarity,BarlowTwinsLoss,VICRegLoss, NTXentLoss,DCLLoss 
import torch 
import torch.nn.functional as F
from torch import Tensor
 
def get_loss(name):
    
    if isinstance(name, dict):

        return Magic_Cube(name)
    
    elif name=="VICRegLoss":
        return VICRegLoss()
    
    elif name=="BarlowTwinsLoss":
        return BarlowTwinsLoss()
    
    elif name=="NegativeCosineSimilarity":
        return NegativeCosineSimilarity()
    
    elif name=="SimCLR":
        return NTXentLoss()
    
    elif name=="MoCo":
        return NTXentLoss(memory_bank_size=4096)
    
    elif name=="DCL":
        return DCLLoss()
        
    elif name=="Var_Fea_2dBV":
        return Var_Fea_2dBV()  
    
    elif name=="VarBatch_2dFV":  #VarianceAlongFeaLoss  VarianceAlongBatchLoss  VarianceAlongViewLoss
        return VarBatch_2dFV()      
    
    elif name=="VarView_2dBF":
        return VarView_2dBF()      
    
    elif name=="CovFea_1dBV":  #VarianceAlongFeaLoss  VarianceAlongBatchLoss  VarianceAlongViewLoss
        return CovFea_1dBV()      
    
    elif name=="CovBatch_1dFV":
        return CovBatch_1dFV()      
    
    elif name=="CovView_1dBF":
        return CovView_1dBF()  
    
    else:
        raise ValueError(f"Unknown loss name: {name}")


class Magic_Cube(torch.nn.Module):
    
    def __init__(self, weights):
        super(Magic_Cube, self).__init__()
        self.weights = weights
        
    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the variance along feature loss.

        Args:
            zs:
                Tensor with shape (num_views, batch_size, num_features).
        """
        loss = 0
        for key, value in self.weights.items():
            temp_loss = get_loss(key)
            loss += value * temp_loss(zs) 
        return loss
class Var_Fea_2dBV(torch.nn.Module):
    """Implementation of variance along feature loss."""

    def __init__(self, eps: float = 0.0001):
        super(Var_Fea_2dBV, self).__init__()
        self.eps = eps

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the variance along feature loss.

        Args:
            zs:
                Tensor with shape (num_views, batch_size, num_features).
        """
        num_views = zs.size(0)
        batch_size = zs.size(1)
        num_features = zs.size(2)

        zs = zs - zs.mean(dim=2, keepdim=True)
        std = torch.sqrt(zs.var(dim=2) + self.eps)
        loss = torch.mean(F.relu(1.0 - std))
 
        return loss


class VarBatch_2dFV(torch.nn.Module):
    """Implementation of variance along batch loss."""

    def __init__(self, eps: float = 0.0001):
        super(VarBatch_2dFV, self).__init__()
        self.eps = eps

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the variance along batch loss.

        Args:
            zs:
                Tensor with shape (num_views, batch_size, num_features).
        """
        num_views = zs.size(0)
        batch_size = zs.size(1)
        num_features = zs.size(2)

        zs = zs - zs.mean(dim=1, keepdim=True)
        std = torch.sqrt(zs.var(dim=1) + self.eps)
        loss = torch.mean(F.relu(1.0 - std))

        return loss


class VarView_2dBF(torch.nn.Module):
    """Implementation of view variance along feature loss."""

    def __init__(self, eps: float = 0.0001):
        super(VarView_2dBF, self).__init__()
        self.eps = eps

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the view variance along feature loss.

        Args:
            zs:
                Tensor with shape (num_views, batch_size, num_features).
        """
        num_views = zs.size(0)
        batch_size = zs.size(1)
        num_features = zs.size(2)

        zs = zs - zs.mean(dim=0, keepdim=True)
        std = torch.sqrt(zs.var(dim=0) + self.eps)
        loss = torch.mean(std)
        return loss
    
    
class CovFea_1dBV(torch.nn.Module):  #CovFea_1dBV CovBatch_1dFV CovView_1dBF
    """Implementation of covariance between batch-view combination loss for a given feature."""

    def __init__(self):
        super(CovFea_1dBV, self).__init__()

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the covariance between batch-view combination loss for a given feature.

        Args:
            zs: Tensor with shape (num_views, batch_size, num_features).
        """
        zs = zs.permute(2, 0, 1)  # shape becomes (num_features, num_views, batch_size)
        zs = zs.reshape(zs.shape[0], -1)  # shape becomes (num_features, num_views*batch_size)
        zs = zs - zs.mean(dim=1, keepdim=True)  # Centralize data

        num_features = zs.size(0)
        batch_size = zs.size(1)
        
        # nondiag_mask has shape (batch_size, batch_size) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(batch_size, device=zs.device, dtype=torch.bool)
        
        # cov has shape (num_features, batch_size, batch_size)
        cov = torch.einsum("ij,ik->ijk", zs, zs) / (batch_size - 1)
        
        # apply nondiag_mask to covariance matrix
        cov = cov[:, nondiag_mask]

        loss = cov.pow(2).sum(-1) / (batch_size - 1)

        return loss.mean()  # Average loss across features


class CovBatch_1dFV(torch.nn.Module):  
    """Implementation of covariance between feature-view combination loss for a given batch."""

    def __init__(self):
        super(CovBatch_1dFV, self).__init__()

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the covariance between feature-view combination loss for a given batch.

        Args:
            zs: Tensor with shape (num_views, batch_size, num_features).
        """
        zs = zs.permute(1, 0, 2)  # shape becomes (batch_size, num_views, num_features)
        zs = zs.reshape(zs.shape[0], -1)  # shape becomes (batch_size, num_views*num_features)
        zs = zs - zs.mean(dim=1, keepdim=True)  # Centralize data

        batch_size = zs.size(0)
        num_variables = zs.size(1)
        
        # nondiag_mask has shape (num_variables, num_variables) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(num_variables, device=zs.device, dtype=torch.bool)
        
        # cov has shape (batch_size, num_variables, num_variables)
        cov = torch.einsum("ij,ik->ijk", zs, zs) / (num_variables - 1)
        
        # apply nondiag_mask to covariance matrix
        cov = cov[:, nondiag_mask]

        loss = cov.pow(2).sum(-1) / (num_variables - 1)

        return loss.mean()  # Average loss across batches


class CovView_1dBF(torch.nn.Module):
    """Implementation of covariance between batch-feature combination loss for a given view."""

    def __init__(self):
        super(CovView_1dBF, self).__init__()

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the covariance between batch-feature combination loss for a given view.

        Args:
            zs: Tensor with shape (num_views, batch_size, num_features).
        """
        zs = zs.permute(0, 2, 1)  # shape becomes (num_views, num_features, batch_size)
        zs = zs.reshape(zs.shape[0], -1)  # shape becomes (num_views, num_features*batch_size)
        zs = zs - zs.mean(dim=1, keepdim=True)  # Centralize data

        num_views = zs.size(0)
        num_variables = zs.size(1)
        
        # nondiag_mask has shape (num_variables, num_variables) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(num_variables, device=zs.device, dtype=torch.bool)
        
        # cov has shape (num_views, num_variables, num_variables)
        cov = torch.einsum("ij,ik->ijk", zs, zs) / (num_variables - 1)
        
        # apply nondiag_mask to covariance matrix
        cov = cov[:, nondiag_mask]

        loss = cov.pow(2).sum(-1) / (num_variables - 1)

        return -loss.mean()  # Average loss across views

class MSEView_1dBF(torch.nn.Module):
    """Implementation of covariance between batch-feature combination loss for a given view."""

    def __init__(self):
        super(CovView_1dBF, self).__init__()

    def forward(self, zs: torch.Tensor) -> torch.Tensor:
        """Returns the covariance between batch-feature combination loss for a given view.

        Args:
            zs: Tensor with shape (num_views, batch_size, num_features).
        """
        zs = zs.permute(0, 2, 1)  # shape becomes (num_views, num_features, batch_size)
        zs = zs.reshape(zs.shape[0], -1)  # shape becomes (num_views, num_features*batch_size)
        zs = zs - zs.mean(dim=1, keepdim=True)  # Centralize data

        num_views = zs.size(0)
        num_variables = zs.size(1)
        
        # nondiag_mask has shape (num_variables, num_variables) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(num_variables, device=zs.device, dtype=torch.bool)
        
        # cov has shape (num_views, num_variables, num_variables)
        cov = torch.einsum("ij,ik->ijk", zs, zs) / (num_variables - 1)
        
        # apply nondiag_mask to covariance matrix
        cov = cov[:, nondiag_mask]

        loss = cov.pow(2).sum(-1) / (num_variables - 1)

        return -loss.mean()  # Average loss across views
