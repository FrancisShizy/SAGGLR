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

        ##### GNN training #####
        for (train_loss, train_reg, train_reg_sparse), setting_name, folder_path in zip(loss_settings, loss_settings_name, folder_path_list):
            print(f"loss setting: {setting_name}")
            print(f"Training process begins: ")
            model = GNN(
                num_node_features,
                num_edge_features,
                hidden_dim_embed = args.hidden_dim_embed, 
                hidden_dim_gnn = args.hidden_dim_gnn,
                hidden_dim_mlp = args.hidden_dim_mlp,
                hidden_dim_linear = args.hidden_dim_linear,
                mask_dim = args.mask_dim,
                num_layers = args.num_layers,
                conv_name = args.conv,
                pool=args.pool,
            ).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5
            )

            # Early stopping
            last_loss = 100
            patience = 5
            trigger_times = 0
            rmse_test_list = []
            pcc_test_list = []
            rmse_delta_test_list = []
            pcc_delta_test_list = []
            epoch_list = []
            final_epoch = 0

            for epoch in range(1, args.epoch + 1):
                final_epoch += 1
                loss = train_epoch(
                    train_dataloader,
                    model,
                    optimizer,
                    loss_type=train_loss,
                    lambda_ucn=args.lambda_ucn,
                    lambda_cn=args.lambda_cn,
                    lambda_group = args.lambda_group,
                    lambda_MSE = args.lambda_MSE,
                    regularization = train_reg,
                    Sparse = train_reg_sparse,
                )

                rmse_val, pcc_val, rmse_delta_val, pcc_delta_val = test_epoch(val_dataloader, model)

                # drive LR by validation performance
                scheduler.step(rmse_val)
                current_lr = scheduler.optimizer.param_groups[0]["lr"]

                rmse_test, pcc_test, rmse_delta_test, pcc_delta_test  = test_epoch(test_dataloader, model)
                
                # Early stopping
                if epoch > 300:
                    if loss > last_loss:
                        trigger_times += 1
                        print("Trigger Times:", trigger_times)

                        if trigger_times >= patience:
                            print(
                                f"Early stopping after {epoch} epochs!\nStart to test process."
                            )
                            break
                    else:
                        print("trigger times: 0")
                        trigger_times = 0
                last_loss = loss
                rmse_test_list.append(rmse_test)
                pcc_test_list.append(pcc_test)
                rmse_delta_test_list.append(rmse_delta_test)
                pcc_delta_test_list.append(pcc_delta_test)
                epoch_list.append(epoch)
                
            final_test_rmse = rmse_test
            final_test_pcc = pcc_test
            final_test_rmse_delta = rmse_delta_test
            final_test_pcc_delta = pcc_delta_test
            best_rmse_test = min(rmse_test_list)
            best_pcc_test = max(pcc_test_list)
            best_rmse_delta_test = min(rmse_delta_test_list)
            best_pcc_delta_test = max(pcc_delta_test_list)
            best_epoch_rmse_test = epoch_list[rmse_test_list.index(best_rmse_test)]
            best_epoch_pcc_test = epoch_list[pcc_test_list.index(best_pcc_test)]
            best_epoch_rmse_delta_test = epoch_list[rmse_delta_test_list.index(best_rmse_delta_test)]
            best_epoch_pcc_delta_test = epoch_list[pcc_delta_test_list.index(best_pcc_delta_test)]

            loss_metrics[setting_name].append({
                  'final_rmse': final_test_rmse,
                  'final_pcc': final_test_pcc,
                  'final_rmse_delta': final_test_rmse_delta,
                  'final_pcc_delta': final_test_pcc_delta,
                  'best_rmse': best_rmse_test,
                  'best_pcc': best_pcc_test,
                  'best_rmse_delta': best_rmse_delta_test,
                  'best_pcc_delta': best_pcc_delta_test
                })

            ###### Save the model performance metrics on th test data
                
            os.makedirs(args.cv_path_test, exist_ok=True)
            os.makedirs(osp.join(args.cv_path_test, args.target), exist_ok=True)
            os.makedirs(osp.join(args.cv_path_test, args.target, f"{args.conv}_{args.pool}"), exist_ok=True)
            os.makedirs(osp.join(args.cv_path_test, args.target, f"{args.conv}_{args.pool}", folder_path), exist_ok=True)
            model_metrics_path = osp.join(args.cv_path_test, args.target, 
                                          f"{args.conv}_{args.pool}", 
                                          folder_path, 
                                          f"model_test_metrics_gnn_fold_{fold}_{args.conv}_{setting_name}_pool_{args.pool}_hiddenDim_{args.hidden_dim_linear}_finalEpoch_{final_epoch}.csv"
            )
            print(f"Saved folder:{args.cv_path_test}{args.target}/{args.conv}_{args.pool}/{folder_path}")
            res_dict = {
                "Epoch": epoch_list, 
                "Final test rmse": rmse_test_list,
                "Final test pcc": pcc_test_list,
                "Final test rmse delta": rmse_delta_test_list,
                "Final test pcc delta": pcc_delta_test_list,
                "Best epoch for rmse test": best_epoch_rmse_test,
                "Best test rmse": best_rmse_test,
                "Best epoch for test rmse delta": best_epoch_rmse_delta_test,
                "Best test rmse delta": best_rmse_delta_test,
                "Best epoch for test pcc": best_epoch_pcc_test,
                "Best test pcc": best_pcc_test,
                "Best epoch for test pcc delta": best_epoch_pcc_delta_test,
                "Best test pcc delta": best_pcc_delta_test,
            }
            df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
            df.to_csv(model_metrics_path, index=False)
        
    # summarize cross‑validation
    rows = []
    for name, metrics in loss_metrics.items():
        final_rmse = [m['final_rmse'] for m in metrics]
        final_pcc = [m['final_pcc'] for m in metrics]
        best_rmse = [m['best_rmse'] for m in metrics]
        best_pcc = [m['best_pcc'] for m in metrics]
        print(f"{name:20s} CV -> Final RMSE: {np.mean(final_rmse):.3f} ± {np.std(final_rmse):.3f} "
            f"Final PCC: {np.mean(final_pcc):.3f} ± {np.std(final_pcc):.3f} "
            f"Best RMSE: {np.mean(best_rmse):.3f} ± {np.std(best_rmse):.3f} "
            f"Best PCC: {np.mean(best_pcc):.3f} ± {np.std(best_pcc):.3f}")
        rows.append({
            'loss_setting': name,
            'final_rmse_avg': np.mean(final_rmse),
            'final_rmse_std': np.std(final_rmse),
            'final_pcc_avg': np.mean(final_pcc),
            'final_pcc_std': np.std(final_pcc),
            'best_rmse_avg': np.mean(best_rmse),
            'best_rmse_std': np.std(best_rmse),
            'best_pcc_avg': np.mean(best_pcc),
            'best_pcc_std': np.std(best_pcc),
        })
        
    df_summary = pd.DataFrame(rows)
    out_csv = f"{args.cv_path_test}/{args.target}/{args.conv}_{args.pool}/loss_cv_performance_hiddenDim_{args.hidden_dim_linear}_topEpoch_{args.epoch}.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved cross-validation summary to {out_csv}\n")
    

if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()
    args.loss_setting_list = ["MSE+UCN","MSE+N"] # "MSE", "MSE+UCN", "MSE+N"
    main_cv(args)
