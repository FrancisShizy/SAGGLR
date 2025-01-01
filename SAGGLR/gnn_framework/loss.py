import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from SAGGLR.utils.utils import (
    get_common_nodes,
    get_positions,
    get_substituent_info,
    get_substituents,
    get_nodes,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_uncommon_node(
    data_i: Tensor, data_j: Tensor, model: nn.Module, reduction: str = "mean"
) -> Tensor:
    """
    Compute the loss that correlates decoration embeddings and activity cliff.
    
    """
    emb_i = model.get_uncommon_graph_rep(data_i)
    emb_j = model.get_uncommon_graph_rep(data_j)
    delta_emb = torch.squeeze(emb_i) - torch.squeeze(emb_j)

    delta_y = data_i.y - data_j.y
    if delta_emb.dim() == 0:
        delta_emb = delta_emb.unsqueeze(0)
    loss = F.mse_loss(delta_emb, delta_y, reduction=reduction)
    return loss


def loss_uncommon_node_local(
    data_i: Tensor, data_j: Tensor, model: nn.Module, reduction: str = "mean"
) -> Tensor:
    """
    Compute the loss that correlates decoration embeddings and activity cliff at individual site of the decorations.
    get_common_nodes: Get the common nodes between two graphs and their mapping to the same indexing
    get_substituents: Get the substituents and the active sites of the scaffold.
    get_positions: Map active sites on the scaffold to the index of the substituent attached to it.
    """
    idx_common_i, idx_common_j, map_i, map_j = get_common_nodes(
        data_i.mask.cpu(), data_j.mask.cpu()
    )

    subs_i, as_i = get_substituents(data_i, idx_common_i, map_i)
    pos_sub_i = get_positions(subs_i, idx_common_i, map_i)

    subs_j, as_j = get_substituents(data_j, idx_common_j, map_j)
    pos_sub_j = get_positions(subs_j, idx_common_j, map_j)

    cmn_sites = np.unique(np.concatenate([as_i, as_j]))
    loss = []
    pos_sub_filtered_i = {k: v for k, v in pos_sub_i.items() if k in cmn_sites}
    pos_sub_filtered_j = {k: v for k, v in pos_sub_j.items() if k in cmn_sites}
    
    for site in cmn_sites:

        if pos_sub_filtered_i[site] == -1:
            emb_i = torch.zeros(1).to(data_i.x.device)
        else:
            sub = subs_i[pos_sub_filtered_i[site]]
            emb_i = model.get_substituent_rep(sub, data_i)

        if pos_sub_filtered_j[site] == -1:
            emb_j = torch.zeros(1).to(data_j.x.device)
        else:
            sub = subs_j[pos_sub_filtered_j[site]]
            emb_j = model.get_substituent_rep(sub, data_j)

        batch_i, a_i = get_substituent_info(site, data_i, map_i)
        batch_j, a_j = get_substituent_info(site, data_j, map_j)
        
        loss.append(
            F.mse_loss(
                torch.squeeze(emb_i - emb_j), a_i - a_j, reduction=reduction
            ).item()
        )
    return np.mean(loss), len(cmn_sites)


def loss_activity_cliff(
    input_i: Tensor,
    input_j: Tensor,
    target_i: Tensor,
    target_j: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute the loss that preserves the activity cliff.
    """
    loss = F.mse_loss(input_i - input_j, target_i - target_j, reduction=reduction)
    return loss


def group_lasso_penalty(model, group_params, lambda_group_lasso):
    penalty = 0
    for name, param in model.named_parameters():
        if name in group_params:
            # penalty += torch.norm(param, p=2)
            if param.ndim > 1: 
                pl = torch.tensor(param.size(1), dtype=torch.float).to(DEVICE)
            else:
                pl = torch.tensor(param.size(0), dtype=torch.float).to(DEVICE)
            penalty += pl.sqrt() * torch.norm(param, p=2)
    return lambda_group_lasso * penalty


def sparse_group_lasso_penalty(model, group_params, lambda_group_lasso, alpha):
    penalty = 0
    for name, param in model.named_parameters():
        if name in group_params:
            if param.ndim > 1: 
                pl = torch.tensor(param.size(1), dtype=torch.float).to(DEVICE)
            else:
                pl = torch.tensor(param.size(0), dtype=torch.float).to(DEVICE)
            penalty += (1-alpha) * lambda_group_lasso * torch.sqrt(pl) * torch.norm(param, 2)  # L2 penalty
    return penalty


def loss_node_with_group_lasso(data_i: Tensor, data_j: Tensor, model: nn.Module, lambda_group_lasso, 
                               common_group_params, uncommon_group_params, reduction: str="mean") -> Tensor:
    """
    compute the loss that coreelates both common and uncommon decoration embeddings with the activity cliff.
    """
    emb_i_uncommon, emb_i_common = model.get_graph_representations(data_i, is_pred = True)
    emb_j_uncommon, emb_j_common = model.get_graph_representations(data_j, is_pred = True)
    
    # Compute deltas for both uncommon and common represenations
    delta_emb_uncommon = torch.squeeze(emb_i_uncommon) - torch.squeeze(emb_j_uncommon)
    delta_emb_common = torch.squeeze(emb_i_common) - torch.squeeze(emb_j_common)
    
    # Ensure they are not zero-dimensional.
    if delta_emb_uncommon.dim() == 0:
        delta_emb_uncommon = delta_emb_uncommon.unsqueeze(0)
    if delta_emb_common.dim() == 0:
        delta_emb_common = delta_emb_common.unsqueeze(0)
    
    # Compute the delta for the target values.
    delta_y = data_i.y - data_j.y
    
    # Compute the loss for both uncommon and common representations.
    loss_uncommon = F.mse_loss(delta_emb_uncommon, delta_y, reduction=reduction)
    loss_common = F.mse_loss(delta_emb_common, delta_y, reduction=reduction)
    
    # Compute the group lasso penalties.
    group_lasso_common = group_lasso_penalty(model, common_group_params, lambda_group_lasso)
    group_lasso_uncommon = group_lasso_penalty(model, uncommon_group_params, lambda_group_lasso)

    # Combine the losses.
    loss = loss_uncommon + loss_common + group_lasso_common + group_lasso_uncommon
    
    return loss
     

def lasso_regular_penalty(model, param_name):
    lasso_reg = torch.tensor(0.).to(DEVICE)
    for name, param in model.named_parameters():
        if name in param_name:
            lasso_reg += torch.norm(param, 1)
    return lasso_reg


def loss_node_with_sparse_group_lasso(data_i: Tensor, data_j: Tensor, model: nn.Module, lambda_group_lasso,
                                      common_group_params, uncommon_group_params, alpha, reduction: str="mean") -> Tensor:
    """
    compute the loss that coreelates both common and uncommon decoration embeddings with the activity cliff.
    """
    emb_i_uncommon, emb_i_common = model.get_graph_representations(data_i, is_pred = True)
    emb_j_uncommon, emb_j_common = model.get_graph_representations(data_j, is_pred = True)
    
    # Compute deltas for both uncommon and common represenations.
    delta_emb_uncommon = torch.squeeze(emb_i_uncommon) - torch.squeeze(emb_j_uncommon)
    delta_emb_common = torch.squeeze(emb_i_common) - torch.squeeze(emb_j_common)
    
    # Ensure they are not zero-dimensional.
    if delta_emb_uncommon.dim() == 0:
        delta_emb_uncommon = delta_emb_uncommon.unsqueeze(0)
    if delta_emb_common.dim() == 0:
        delta_emb_common = delta_emb_common.unsqueeze(0)
    
    # Compute the delta for the target values.
    delta_y = data_i.y - data_j.y
    
    # Compute the loss for both uncommon and common representations.
    loss_uncommon = F.mse_loss(delta_emb_uncommon, delta_y, reduction=reduction)
    loss_common = F.mse_loss(delta_emb_common, delta_y, reduction=reduction)
    
    # Compute the group lasso penalties.
    sparse_group_lasso_common = sparse_group_lasso_penalty(model, common_group_params, lambda_group_lasso, alpha)
    sparse_group_lasso_uncommon = sparse_group_lasso_penalty(model, uncommon_group_params, lambda_group_lasso, alpha)

    # Combine the losses (you might want to weigh them differently)
    loss = loss_uncommon + loss_common + sparse_group_lasso_common + sparse_group_lasso_uncommon
    
    return loss
     


def loss_node_local(
    data_i: Tensor, data_j: Tensor, model: nn.Module, reduction: str = "mean"
) -> Tensor:
    """
    Compute the loss that correlates decoration embeddings and activity cliff at individual site of the decorations.
    Compute the loss for the embbeding prediction difference between the common sites for molecule i and molecule j and then calculate the mean.
    
    The fucntions includes:
    get_nodes: Get the common and uncommon nodes between two graphs and their mapping to the same indexing.
    get_substituents: Get the substituents and the active sites of the scaffold.
    get_positions: Map active sites on the scaffold to the index of the substituent attached to it.
    """
    idx_common_i, idx_common_j, idx_uncommon_i, idx_uncommon_j, map_i, map_j = get_nodes(
        data_i.mask.cpu(), data_j.mask.cpu()
    )

    subs_i, as_i = get_substituents(data_i, idx_common_i, map_i)
    pos_sub_i = get_positions(subs_i, idx_common_i, map_i)

    subs_j, as_j = get_substituents(data_j, idx_common_j, map_j)
    pos_sub_j = get_positions(subs_j, idx_common_j, map_j)

    cmn_sites = np.unique(np.concatenate([as_i, as_j]))
    loss = []
    pos_sub_filtered_i = {k: v for k, v in pos_sub_i.items() if k in cmn_sites}
    pos_sub_filtered_j = {k: v for k, v in pos_sub_j.items() if k in cmn_sites}
    
    
    for site in cmn_sites:

        if pos_sub_filtered_i[site] == -1:
            emb_i = torch.zeros(1).to(data_i.x.device)
        else:
            sub = subs_i[pos_sub_filtered_i[site]]
            emb_i = model.get_substituent_rep(sub, data_i)

        if pos_sub_filtered_j[site] == -1:
            emb_j = torch.zeros(1).to(data_j.x.device)
        else:
            sub = subs_j[pos_sub_filtered_j[site]]
            emb_j = model.get_substituent_rep(sub, data_j)

        batch_i, a_i = get_substituent_info(site, data_i, map_i)
        batch_j, a_j = get_substituent_info(site, data_j, map_j)

        loss.append(
            F.mse_loss(
                torch.squeeze(emb_i - emb_j), a_i - a_j, reduction=reduction
            ).item()
        )
        
    #print(idx_common_i)
    #print(idx_uncommon_i)    
    #print(subs_i)
    #print(subs_j)
    #print(pos_sub_i)
    #print(pos_sub_j)
    #print(cmn_sites)
    #print(pos_sub_filtered_i)
    #print(pos_sub_filtered_j)
    #print(sub)
    #print(site)
    #print(emb_i)
    #print(emb_j)
    #print(as_i)
    #print(as_j)
    #print(loss)
    #sys.exit()
    return np.mean(loss), len(cmn_sites)
