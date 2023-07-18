
def Cov_batch_ik_10(x: torch.Tensor) -> torch.Tensor:
    # Consider only the first two views
    x = x[:2]

    # Calculate the covariance matrices for each view
    cov_matrices = [torch.cov(view.T) for view in x]

    # Zero out the diagonal elements of each covariance matrix
    offdiag_elements = [cov_matrix - torch.diag(torch.diag(cov_matrix)) for cov_matrix in cov_matrices]

    # Flatten and average the off-diagonal elements
    average_offdiag = torch.cat([elem.flatten() for elem in offdiag_elements]).mean()

    return average_offdiag


def Cov_batch_ik_01(x: torch.Tensor) -> torch.Tensor:
    x = x[:2]  # considering only two views
    view_1, view_2 = x[0], x[1]
    mean_view_1 = torch.mean(view_1, dim=0)
    mean_view_2 = torch.mean(view_2, dim=0)
    covars = [(view_1[:, i] - mean_view_1[i]) * (view_2[:, i] - mean_view_2[i]) for i in range(view_1.shape[1])]
    covars = [torch.mean(covar) for covar in covars]
    return torch.tensor(covars)

def Cov_batch_ik_00(x: torch.Tensor) -> torch.Tensor:
    x = x[:2]  # considering only two views
    view_1, view_2 = x[0], x[1]
    mean_view_1 = torch.mean(view_1, dim=0)
    mean_view_2 = torch.mean(view_2, dim=0)
    off_diag_covars = []
    for i in range(view_1.shape[1]):
        for j in range(i+1, view_1.shape[1]):
            off_diag_covar = (view_1[:, i] - mean_view_1[i]) * (view_2[:, j] - mean_view_2[j])
            off_diag_covars.append(torch.mean(off_diag_covar))
    return torch.tensor(off_diag_covars)


# Use the data with your functions:
result_10 = Cov_batch_ik_10(x)
result_01 = Cov_batch_ik_01(x)
result_00 = Cov_batch_ik_00(x)

print("Result 10:", result_10)
print("Result 01:", result_01)
print("Result 00:", result_00)
print("Result 1111:", covariance_loss(x))



def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    res = 0
    for i in range(2):
        view = x[i]
        view = view - view.mean(dim=0)
        batch_size = view.size(0)
        dim = view.size(-1)
        # nondiag_mask has shape (dim, dim) with 1s on all non-diagonal entries.
        nondiag_mask = ~torch.eye(dim, device=view.device, dtype=torch.bool)
        # cov has shape (..., dim, dim)
        cov = torch.einsum("bc,bd->cd", view, view) / (batch_size - 1)
        loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim

        res += loss.mean()
    return res


from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

pdata= PipeDataset(config=config,augmentation=trans2multi(SSL_augmentation,view=config["view"]))
pmodel=pipe_model(config=config,MonitoringbyKNN=p_knndata) # All save the validation

y = pmodel(next(iter(pdata.dataloader))[0][0])
make_dot(y, params=dict(pmodel.named_parameters()))
