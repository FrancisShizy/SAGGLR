from functools import wraps
import sys
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from SAGGLR.gnn_framework.loss import loss_uncommon_node, loss_common_node, lasso_regular_penalty, loss_node_with_group_lasso, loss_node_local, loss_node_with_sparse_group_lasso
from SAGGLR.utils.device import DEVICE

MSE_LOSS_FN = nn.MSELoss()

def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim,
    loss_type: str = "MSE",
    lambda1: int = 1,
    lambda_group = 0.001,
    alpha = 0.5,
    lambda_MSE = 0.001,
    regularization = True,
    Sparse = True,
    common_group_params = ['lin_common_layer.weight', 'lin_common_layer.bias'],  # Parameters for common subgraph processing.
    uncommon_group_params = ['lin_uncommon_layer.weight', 'lin_uncommon_layer.bias'],  # Parameters for uncommon subgraph processing.
    final_layer_params = ['final_layer.weight', 'final_layer.bias'],
):
    """Train the model for one epoch."""
    model.train()
    loss_all = 0

    progress = tqdm(train_loader)
    total = 0
    for hetero_data in progress:
        data_i = Batch().from_data_list(hetero_data["data_i"])
        data_j = Batch().from_data_list(hetero_data["data_j"])
        optimizer.zero_grad()
        data_i, data_j = data_i.to(DEVICE), data_j.to(DEVICE)
        out_i, out_j = torch.squeeze(model(data_i)), torch.squeeze(model(data_j))
        if out_i.dim() == 0:
            out_i = out_i.unsqueeze(0)
        if out_j.dim() == 0:
            out_j = out_j.unsqueeze(0)

        mse_i, mse_j = MSE_LOSS_FN(out_i, data_i.y.to(DEVICE)), MSE_LOSS_FN(
            out_j, data_j.y.to(DEVICE)
        )
        loss = (mse_i + mse_j) / 2

        # Additional loss calculations based on loss_type
        if loss_type == "MSE": # only considering MSE of predicted vs. true activity scores
            if regularization:
                loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)
            else:
                pass
        else:
            if loss_type == "MSE+UCN": # considering MSE & uncommon node information
                loss += lambda1 * loss_uncommon_node(data_i, data_j, model)
            elif loss_type == "MSE+N": # considering MSE & all node information
                if regularization:
                    if Sparse: # sparse group lasso 
                        loss += loss_node_with_sparse_group_lasso(data_i, data_j, model, lambda_group, common_group_params, uncommon_group_params, alpha)
                        loss += alpha * lambda_group * lasso_regular_penalty(model, common_group_params)
                        loss += alpha * lambda_group * lasso_regular_penalty(model, uncommon_group_params)
                        # loss += lambda_group * lasso_regular_penalty(model, final_layer_params)
                    else: # group lasso
                        loss += loss_node_with_group_lasso(data_i, data_j, model, lambda_group, common_group_params, uncommon_group_params)
                else: # no group lasso
                    loss += lambda1 * loss_common_node(data_i, data_j, model)
            elif loss_type == "MSE+AC":
                    loss += MSE_LOSS_FN(
                        out_i - out_j, data_i.y.to(DEVICE) - data_j.y.to(DEVICE)
                    )
                    if regularization:
                        loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)
            elif loss_type == "MSE+UCNlocal":
                    loss_batch, n_subs_in_batch = loss_node_local(
                        data_i, data_j, model
                    )
                    loss += lambda1 * loss_batch
                    # argue whether to use regularization
                    if regularization:
                        loss += lambda_MSE * lasso_regular_penalty(model, final_layer_params)

                    loss.backward()
                    loss_all += loss.item() * n_subs_in_batch

                    optimizer.step()
                    progress.set_postfix({"loss": loss.item()})
                    total += n_subs_in_batch
            else: 
                AssertionError("Please provide correct loss fuctions from MSE, MSE+UCN, MSE+N, or MSE+AC, etc.")
        
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * (data_i.num_graphs if hasattr(data_i, 'num_graphs') else 1)
        progress.set_postfix({"loss": loss.item()})
        total += (data_i.num_graphs if hasattr(data_i, 'num_graphs') else 1)

    return loss_all / total


def test_epoch(test_loader: DataLoader, model: nn.Module):
    """Evaluates the trained GNN model on the testing pairs with RMSE and PCC metrics."""

    model.eval()
    error = 0
    pcc = 0
    with torch.no_grad():
        k = 0
        for hetero_data in test_loader:
            data_i = Batch().from_data_list(hetero_data["data_i"]).to(DEVICE)
            data_j = Batch().from_data_list(hetero_data["data_j"]).to(DEVICE)
            out_i, out_j = model(data_i), model(data_j)
            rmse_i = torch.sqrt(MSE_LOSS_FN(torch.squeeze(out_i).detach(), data_i.y)).item()
            rmse_j = torch.sqrt(MSE_LOSS_FN(torch.squeeze(out_j).detach(), data_j.y)).item()
            error += (rmse_i + rmse_j) / 2 * hetero_data.num_graphs

            pcc_i = np.corrcoef(torch.squeeze(out_i).cpu().numpy(), data_i.y.cpu().numpy())[0, 1]
            pcc_j = np.corrcoef(torch.squeeze(out_j).cpu().numpy(), data_j.y.cpu().numpy())[0, 1]
            if np.isnan(pcc_i) | np.isnan(pcc_j):
                try:
                    k += 1
                    raise ValueError(
                        "PCC is NaN. The batch has the same compound for all pairs. This batch will be ignored for the final results computation.\n \
                                    If you want to avoid removing batches, try shuffling the pair set with new seed."
                    )
                except ValueError as err:
                    print(err)
            else:
                pcc += (pcc_i + pcc_j) / 2 * hetero_data.num_graphs

        return error / len(test_loader.dataset), pcc / (len(test_loader.dataset) - k)



def overload(func):
    """This function is used in the model class to overload the forward function."""

    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) + len(kargs) == 2:
            if len(args) == 2:  # for inputs like model(g)
                g = args[1]
            else:  # for inputs like model(graph=g)
                g = kargs["graph"]
            return func(args[0], g)

        elif len(args) + len(kargs) == 5:
            if len(args) == 5:  # for inputs like model(x, ..., batch)
                return func(*args)
            else:  # for inputs like model(x=x, ..., batch=batch)
                return func(args[0], **kargs)

        elif len(args) + len(kargs) == 6:
            if len(args) == 6:  # for inputs like model(x, ..., batch, pos)
                return func(*args[:-1])
            else:  # for inputs like model(x=x, ..., batch=batch, pos=pos)
                return func(
                    args[0],
                    kargs["x"],
                    kargs["edge_index"],
                    kargs["edge_attr"],
                    kargs["batch"],
                )
        else:
            raise TypeError

    return wrapper
