import os
import os.path as osp
import time
import sys
import dill
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from data_preprocess.data_split import rebalance_pairs, create_ligands_list
from SAGGLR.gnn_framework.model import GNN
from SAGGLR.utils.parser_utils import overall_parser
from SAGGLR.utils.train_utils import DEVICE, test_epoch, train_epoch
from SAGGLR.utils.utils import get_num_features, get_colors, get_mcs, set_seed
from data_preprocess.scaffold_analysis import analyze_from_pairs_list, leakage_from_split_pairs


def pairs_to_df_from_list(pairs):
    rows = []
    for p in pairs:
        rows.append((p["data_i"].smiles, p["data_j"].smiles))
    return pd.DataFrame(rows, columns=["smiles_i", "smiles_j"])


def main_cv(args):
    set_seed(args.seed)
    torch.cuda.empty_cache()

    ##### Data loading and pre-processing #####
    print(f"{args.data_path}/{args.target}")

    # Check that data exists
    file_path = osp.join(
        args.data_path, f"{args.target}/{args.target}_heterodata_list.pt"
    )
    
    if not osp.exists(file_path):
        raise FileNotFoundError(
            "Data not found. Please try to - choose another protein target."
        )

    with open(file_path, "rb") as handle:
        pairs_list = dill.load(handle)
    n = len(pairs_list)
    ligands_list = create_ligands_list(pairs_list)

    # Loss & regilarization combinations 
    loss_settings = []
    loss_settings_name = []
    folder_path_list = []
    for loss in args.loss_setting_list:
        if loss == "MSE":
            reg = False
            Sparse = False
            loss_settings.append([loss, reg, Sparse]) # L_MSE
            loss_settings_name.append("L_MSE")
            train_params = (f"{args.conv}_L_MSE_{args.pool}_hiddenDim_{args.hidden_dim_linear}")
            folder_path_list.append("MSE")
        
        elif loss == "MSE+UCN":
            reg = False
            Sparse = False
            loss_settings.append([loss, reg, Sparse]) # L_MSE+UCN
            loss_settings_name.append("L_MSE_UCN")
            train_params = (f"{args.conv}_L_MSE_UCN_{args.pool}_hiddenDim_{args.hidden_dim_linear}")
            folder_path_list.append("MSE_UCN")

        elif loss == "MSE+N":
            for reg in [False, True]:
                if reg:
                    for Sparse in [False, True]:
                        if Sparse:
                            loss_settings.append([loss, reg, Sparse]) # L_{MSE+N} + SGL
                            loss_settings_name.append("L_MSE_N_SGL")
                            folder_path_list.append("MSE_N_sparse_group_lasso")
                        else:
                            loss_settings.append([loss, reg, Sparse]) # L_{MSE+N} + GL
                            loss_settings_name.append("L_MSE_N_GL")
                            folder_path_list.append("MSE_N_group_lasso")
                else:
                    Sparse = False
                    loss_settings.append([loss, reg, Sparse]) # L_{MSE+N}
                    loss_settings_name.append("L_MSE_N")
                    folder_path_list.append("MSE_N")

    # Save metrics as a folder
    loss_metrics = {f"{settings}": [] for settings in loss_settings_name}
    
    # Cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(ligands_list), 1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")
        test_set_fraction = 1/args.k_folds

        # Split pairs_list intoi train_pairs and test_pairs sets
        train_ligands = [ligands_list[i] for i in train_idx]
        test_ligands = [ligands_list[i] for i in test_idx]
        train_pairs, test_pairs = [], []
        for pair in pairs_list:
            if (pair["data_i"].smiles in train_ligands) and (
                pair["data_i"].smiles in train_ligands
            ):
                train_pairs.append(pair)
            elif (pair["data_i"].smiles in test_ligands) and (
                pair["data_i"].smiles in test_ligands
            ):
                test_pairs.append(pair)
        train_pairs_dataset, test_pairs_dataset = rebalance_pairs(train_pairs, test_pairs, test_set_fraction)
        # print(len(train_pairs_dataset))
        # print(len(test_pairs_dataset))

        # carve off 10% of the training indices for validation
        train_dataset, val_dataset = train_test_split(
            train_pairs_dataset,
            test_size=args.val_set_size,
            random_state=args.seed,
            shuffle=True,
        )
        # print(len(train_dataset))
        # print(len(val_dataset))
        
        # Load data using DataLoader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_dataloader = DataLoader(
            test_pairs_dataset,
            batch_size=len(test_pairs_dataset),
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_node_features, num_edge_features = get_num_features(train_dataset[0])

        # Save scaffold analysis per fold
        fold_dir = osp.join(args.cv_scaffold_analysis_root, args.target, "scaffold_analysis", f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Analyze each split
        analyze_from_pairs_list(train_dataset, fold_dir, prefix="train_")
        analyze_from_pairs_list(val_dataset, fold_dir, prefix="val_")
        analyze_from_pairs_list(test_pairs_dataset, fold_dir, prefix="test_")

        # Overlap / leakage report
        leakage_from_split_pairs(
            train_pairs_df=pairs_to_df_from_list(train_dataset),
            val_pairs_df=pairs_to_df_from_list(val_dataset),
            test_pairs_df=pairs_to_df_from_list(test_pairs_dataset),
            out_dir=fold_dir,
            prefix="",
        )
    

if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()
    args.loss_setting_list = ["MSE+UCN","MSE+N"] # "MSE", "MSE+UCN", "MSE+N"
    main_cv(args)
