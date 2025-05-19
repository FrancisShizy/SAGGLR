from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import ModuleList, ReLU
from torch_geometric.nn import NNConv, GINEConv, GATConv, GENConv
from torch_geometric.nn import (
    global_mean_pool, global_max_pool, global_add_pool,
    AttentionalAggregation, BatchNorm
)
from SAGGLR.utils.train_utils import overload


class GNN(nn.Module):
    """
    Graph Neural Network for node classification/regression tasks.

    Args:
        num_node_features (int): Size of node feature vectors.
        num_edge_features (int): Size of edge feature vectors.
        hidden_dim (int): Hidden dimension for embeddings and layers.
        mask_dim (int): Dimension for mask layers.
        num_layers (int): Number of graph convolution layers.
        num_classes (int): Output dimension (number of target classes).
        conv_name (str): Type of graph convolution ('nn', 'gine', 'gat', 'gen').
        pool (str): Global pooling method ('mean', 'max', 'add', 'att', 'mean+att').
    """
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 32,
        mask_dim: int = 16,
        num_layers: int = 2,
        num_classes: int = 1,
        conv_name: str = 'nn',
        pool: str = 'mean'
    ):
        super().__init__()
        
        # Save hyperparameters
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Embedding layers
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        self.edge_emb = nn.Linear(num_edge_features, hidden_dim)

        # Build graph convolution, normalization, and activation blocks
        self.convs, self.batch_norms, self.activations = self._build_conv_blocks(conv_name)

        # Prediction layers for common and uncommon subgraphs
        self.lin_common_pred   = nn.Linear(hidden_dim, num_classes)
        self.lin_uncommon_pred = nn.Linear(hidden_dim, num_classes)

        # Masking layers
        self.lin_common_layer   = nn.Linear(hidden_dim, mask_dim)
        self.lin_uncommon_layer = nn.Linear(hidden_dim, mask_dim)
        self.final_layer        = nn.Linear(2 * mask_dim, num_classes)

        # Uncommon-only path prediction
        self.lin1 = nn.Linear(hidden_dim, num_classes)

        # Pooling setup
        self.pool, self.pool_fn = self._get_pool_fn(pool)

    def _build_conv_blocks(self, conv_name: str):
        """
        Create graph convolution layers with matching normalization and activation.
        """
        convs = nn.ModuleList()
        norms = nn.ModuleList()
        activations = nn.ModuleList()
        for _ in range(self.num_layers):
            convs.append(self._create_conv_layer(conv_name))
            norms.append(BatchNorm(self.hidden_dim))
            activations.append(nn.ReLU())

        return convs, norms, activations

    def _create_conv_layer(self, conv_name: str) -> nn.Module:
        """
        Return a graph convolution layer based on conv_name.
        """
        if conv_name == 'nn':
            gate_nn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)
            )
            return NNConv(self.hidden_dim, self.hidden_dim, nn=gate_nn)
        elif conv_name == 'gine':
            return GINEConv(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * self.hidden_dim, self.hidden_dim)
                )
            )
        elif conv_name == 'gat':
            return GATConv(self.hidden_dim, self.hidden_dim, edge_dim=self.hidden_dim, concat=False)
        elif conv_name == 'gen':
            return GENConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError(f"Unsupported conv_name: {conv_name}")

    def _get_pool_fn(self, pool: str):
        """
        Return the appropriate pooling function (and supplementary for 'mean+att').
        """
        if pool == 'att':
            pool_fn = AttentionalAggregation(
                gate_nn=nn.Sequential(nn.Linear(self.hidden_dim, 1)),
                nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
            )
        elif pool == 'mean':
            pool_fn = global_mean_pool
        elif pool == 'max':
            pool_fn = global_max_pool
        elif pool == 'add':
            pool_fn = global_add_pool
        elif pool == 'mean+att':
            pool_fn = global_mean_pool
            self.pool_fn_ucn = AttentionalAggregation(
                gate_nn=nn.Sequential(nn.Linear(self.hidden_dim, 1)),
                nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim))
            )
        else:
            raise ValueError(f"Unsupported pool: {pool}")
        return pool, pool_fn


    @overload
    def forward(self, data: torch.Tensor, is_pred = False):
        # Get embeddings of uncommon and common substructures:
        emb_uncommon, emb_common = self.get_graph_representations(data, is_pred)

        # Combine embeddings
        combined_emb = torch.cat([emb_uncommon, emb_common], dim=1)

        # Final prediction for the whole graph
        final_pred = self.final_layer(combined_emb)
        return final_pred


    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        """Returns the node embeddings just before the pooling layer."""
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.activations):
            x = conv(x, edge_index, edge_attr)
            x = relu(batch_norm(x))
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, data):
        # def get_graph_rep(self, x, edge_index, edge_attr, batch):
        """Returns the graph embedding after pooling."""
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        graph_x = self.pool_fn(node_x, data.batch)
        return graph_x

    @overload
    def get_gap_activations(self, data):
        # def get_gap_activations(self, x, edge_index, edge_attr, batch):
        """Returns the node-wise and edge-wise contributions to graph embedding (before the pooling layer)."""
        node_act = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        edge_act = self.edge_emb(data.edge_attr)
        return node_act, edge_act

    def get_prediction_weights(self):
        """Gets prediction weights of the before last layer."""
        # w = self.lin1.weight.data[0]
        w = self.final_layer.weight.data[0].squeeze()
        return w

    @overload
    def get_intermediate_activations_gradients(self, data):
         # def get_intermediate_activations_gradients(self, x, edge_index, edge_attr, batch):
        """Gets intermediate layer activations and gradients."""
        acts = []
        grads = []

        x = Variable(data.x, requires_grad=True)
        edge_attr = Variable(data.edge_attr, requires_grad=True)
        acts.append((x, edge_attr))

        x = self.node_emb(x)
        x.retain_grad()
        edge_attr = self.edge_emb(edge_attr)
        edge_attr.retain_grad()
        acts.append((x, edge_attr))

        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.activations):
            x = conv(x, data.edge_index, edge_attr)
            x = relu(batch_norm(x))
            x.retain_grad()
            edge_attr.retain_grad()
            acts.append((x, edge_attr))
        node_x = x
        graph_x = self.pool_fn(node_x, data.batch)
        y = self.get_pred(graph_x)
        y.backward()
        grads = [(act[0].grad, act[1].grad) for act in acts]
        return acts, grads, y

    def get_substituent_rep(self, sub: List[int], data: torch.Tensor) -> torch.Tensor:
        """Gets the hidden representation of one substituent in a molecule as the model prediction on this subgraph.
        Args:
            sub (List[int]): list of the indicies of the substituent atoms
            data (torch.Tensor): data object containing the graph

        Returns:
            torch.Tensor: hidden representation of the substituent
        """
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        bool_mask = torch.zeros(
            data.mask.size(), dtype=torch.bool, device=data.x.device
        )
        bool_mask[sub] = 1
        masked_batch = data.batch[bool_mask]
        bs = len(torch.unique(data.batch))
        masked_bs = len(torch.unique(masked_batch))
        assert masked_bs == 1
        unique_batch = torch.zeros(
            data.mask.size(), dtype=torch.long, device=data.x.device
        )
        if masked_batch.numel() == 0:
            print("No elements in batch")
            return torch.zeros(bs, self.num_classes).to(data.x.device)
        if self.pool == "att":
            uncommon_graph_x = self.pool_fn.masked_forward(
                node_x, bool_mask, unique_batch
            )
        if self.pool == "mean+att":
            uncommon_graph_x = self.pool_fn_ucn.masked_forward(
                node_x, bool_mask, unique_batch
            )
        else:
            uncommon_graph_x = self.pool_fn(node_x[bool_mask], batch=None)
        uncommon_pred = self.lin1(uncommon_graph_x)
        return uncommon_pred

    def get_uncommon_graph_rep(self, data: torch.Tensor) -> torch.Tensor:
        """Gets the hidden representation of the uncommon part of a molecule as the model prediction on this subgraph.
        Args:
            data (torch.Tensor): data object containing the graph

        Returns:
            torch.Tensor: hidden representation of the uncommon part of the molecule
        """
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        mask = data.mask.cpu().numpy()
        bool_mask = torch.BoolTensor(np.where(mask == 0, 0, 1))
        masked_batch = data.batch[bool_mask]
        bs = len(torch.unique(data.batch))
        masked_bs = len(torch.unique(data.batch[bool_mask]))

        if masked_batch.numel() == 0:
            return torch.zeros(bs, self.num_classes).to(data.x.device)

        if self.pool == "att":
            uncommon_graph_x = self.pool_fn.masked_forward(
                node_x, bool_mask, data.batch
            )
        if self.pool == "mean+att":
            uncommon_graph_x = self.pool_fn_ucn.masked_forward(
                node_x, bool_mask, data.batch
            )
        else:
            uncommon_graph_x = self.pool_fn(node_x[bool_mask], data.batch[bool_mask])

        uncommon_pred = self.lin1(uncommon_graph_x)

        if masked_bs < bs:
            non_zeros_mask_idx = np.intersect1d(
                torch.unique(data.batch).cpu().numpy(),
                torch.unique(data.batch[bool_mask]).cpu().numpy(),
            )
            new_emb = torch.zeros(bs, self.num_classes).to(data.x.device)
            new_emb[non_zeros_mask_idx] = uncommon_pred[non_zeros_mask_idx]
            uncommon_pred = new_emb
        return uncommon_pred

    def get_pred(self, graph_x):
        """Returns the prediction of the model on a graph embedding after the graph convolutional layers."""
        pred = self.lin1(graph_x)
        return pred
    

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)


    def get_graph_representations(self, data: torch.Tensor, is_pred = True):
        """Gets the hidden representation of both the uncommon and common parts of a molecule.
        Args:
            data (torch.Tensor): data object containing the graph

        Returns:
            1. If hope to combine both information of uncommon and common subgraphs, output the mask 
            Tuple[torch.Tensor, torch.Tensor]: hidden representations of the uncommon and common parts of the molecule
            
            2. If output the seperated prediction of uncommon and common subgraphs:
            Tuple[torch.Tensor, torch.Tensor]: hidden representations of the uncommon and common parts of the molecule
        """
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)

        # Masks for uncommon and common graphs
        mask_uncommon = data.mask.cpu().numpy()
        bool_mask_uncommon = torch.BoolTensor(np.where(mask_uncommon == 0, 0, 1))
        bool_mask_common = torch.BoolTensor(np.where(mask_uncommon == 0, 1, 0))

        bs = len(torch.unique(data.batch))
        
        # Process uncommon graph
        masked_batch_uncommon = data.batch[bool_mask_uncommon]
        masked_bs_uncommon = len(torch.unique(masked_batch_uncommon))
        uncommon_mask = self.process_subgraph(node_x, bool_mask_uncommon, masked_batch_uncommon, bs, masked_bs_uncommon, is_uncommon=True, is_pred=is_pred)

        # Process common graph
        masked_batch_common = data.batch[bool_mask_common]
        masked_bs_common = len(torch.unique(masked_batch_common))
        common_mask = self.process_subgraph(node_x, bool_mask_common, masked_batch_common, bs, masked_bs_common, is_uncommon=False, is_pred=is_pred)

        return uncommon_mask, common_mask

    def process_subgraph(self, node_x, bool_mask, masked_batch, bs, masked_bs, is_uncommon=True, is_pred=True):
        """Processes a subgraph to get its hidden representation."""
        if masked_batch.numel() == 0 and is_pred == True:
            return torch.zeros(bs, self.num_classes).to(node_x.device)
        elif masked_batch.numel() == 0 and is_pred == False:
            return torch.zeros(bs, self.mask_dim).to(node_x.device)

        # Pooling and linear transformation
        if self.pool in ["att", "mean+att"]:
            graph_x = self.pool_fn.masked_forward(node_x, bool_mask, masked_batch)
        else:
            graph_x = self.pool_fn(node_x[bool_mask], masked_batch)
        
        # Prediction of linear layers for common and uncommon subgraphs
        if is_pred:    
            if is_uncommon:    
                graph_pred = self.lin_uncommon_pred(graph_x)
            else:
                graph_pred = self.lin_common_pred(graph_x) # nn.Linear(graph_x, self.hidden_dim, self.num_classes).
        else:
            if is_uncommon:    
                graph_pred = self.lin_uncommon_layer(graph_x)
            else:
                graph_pred = self.lin_common_layer(graph_x)
                        

        # Adjust batch sizes if necessary
        if masked_bs < bs:
            non_zeros_mask_idx = np.intersect1d(
                torch.unique(masked_batch).cpu().numpy(),
                torch.unique(masked_batch).cpu().numpy(),
            )
            if is_pred:
                new_emb = torch.zeros(bs, self.num_classes).to(node_x.device)
            else:
                new_emb = torch.zeros(bs, self.mask_dim).to(node_x.device)
            new_emb[non_zeros_mask_idx] = graph_pred[non_zeros_mask_idx]
            graph_pred = new_emb

        return graph_pred
