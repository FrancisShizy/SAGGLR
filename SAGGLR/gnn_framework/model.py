from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import ModuleList, ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import (
    BatchNorm,
    NNConv,
    global_mean_pool,
)
from SAGGLR.utils.train_utils import overload


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 32,
        mask_dim: int = 16,
        num_layers: int = 2,
        num_classes: int = 1,
        conv_name: str = "nn",
        pool: str = "mean",
    ):
        super().__init__()
        (
            self.num_node_features,
            self.num_edge_features,
            self.num_classes,
            self.num_layers,
            self.hidden_dim,
            self.mask_dim,
        ) = (
            num_node_features,
            num_edge_features,
            num_classes,
            num_layers,
            hidden_dim,
            mask_dim,
        )
        
        self.node_emb = nn.Linear(self.num_node_features, self.hidden_dim)
        self.edge_emb = nn.Linear(self.num_edge_features, self.hidden_dim)

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(self.num_layers):
            if conv_name == "nn":
                conv = NNConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    nn = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * self.hidden_dim)),
                )
            else:
                raise ValueError(f"Unknown convolutional layer {conv_name}")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))
            self.relus.append(ReLU())

        # Prediction of nn.Linear layers for common and uncommon subgraphs
        self.lin_common_pred = nn.Linear(self.hidden_dim, self.num_classes)
        self.lin_uncommon_pred = nn.Linear(self.hidden_dim, self.num_classes)   
        
        
        # Separate linear layers for common and uncommon subgraphs
        self.lin_common_layer = nn.Linear(self.hidden_dim, self.mask_dim)
        self.lin_uncommon_layer = nn.Linear(self.hidden_dim, self.mask_dim)
        self.final_layer = nn.Linear(2 * self.mask_dim, self.num_classes)
        # If only use the uncommon nodes
        self.lin1 = nn.Linear(self.hidden_dim, self.num_classes)

        self.pool = pool
        if  self.pool == "mean":
            self.pool_fn = global_mean_pool
        else:
            raise ValueError(f"Unknown pool {self.pool}")


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
        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
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

        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
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
