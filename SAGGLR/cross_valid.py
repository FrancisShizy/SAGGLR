import os
import os.path as osp
import time
import sys
import dill
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from SAGGLR.evaluation.explain_color import get_scores
from SAGGLR.evaluation.explain_direction_global import get_global_directions
from SAGGLR.evaluation.feature_attribution import CAM, GradCAM, GradInput, IntegratedGradient
from SAGGLR.gnn_framework.model import GNN
from SAGGLR.utils.parser_utils import overall_parser
from SAGGLR.utils.train_utils import DEVICE, test_epoch, train_epoch
from SAGGLR.utils.utils import get_num_features, get_colors, get_mcs, set_seed

def main_cv(args):
    set_seed(args.seed)
    torch.cuda.empty_cache()

    ##### Data loading and pre-processing #####
    print(args.data_path)
    print(args.result_path)

    # Check that data exists
    file_path = osp.join(
        args.data_path, f"{args.target}/{args.target}_heterodata_list.pt"
    )

    if not osp.exists(file_path):
        raise FileNotFoundError(
            "Data not found. Please try to - choose another protein target."
        )

    with open(file_path, "rb") as handle:
        all_dataset = dill.load(handle)
    n = len(all_dataset)

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
            train_params = (f"{args.conv}_L_MSE_{args.pool}_hiddenDim_{args.hidden_dim}")
            folder_path_list.append("MSE")
        
        elif loss == "MSE+UCN":
            reg = False
            Sparse = False
            loss_settings.append([loss, reg, Sparse]) # L_MSE+UCN
            loss_settings_name.append("L_MSE_UCN")
            train_params = (f"{args.conv}_L_MSE_UCN_{args.pool}_hiddenDim_{args.hidden_dim}")
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
        elif loss == "MSE+AC":
            reg = False
            Sparse = False
            loss_settings.append([loss, reg, Sparse]) # L_MSE
            loss_settings_name.append("L_MSE_AC")
            train_params = (f"{args.conv}_L_MSE_AC_{args.pool}_hiddenDim_{args.hidden_dim}")
            folder_path_list.append("MSE_AC")

    # Save metrics as a folder
    loss_metrics = {f"{settings}": [] for settings in loss_settings_name}
    
    # Cross-validation
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_dataset), 1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")
        
        # carve off 10% of the training indices for validation
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=args.val_set_size,
            random_state=args.seed,
            shuffle=True,
        )
        
        train_dataset = [all_dataset[i] for i in train_idx]
        val_dataset   = [all_dataset[i] for i in val_idx]
        test_dataset = [all_dataset[i] for i in test_idx]

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,  # num_workers can go into args
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,  # num_workers can go into args
        )

        num_node_features, num_edge_features = get_num_features(train_dataset[0])

        ##### GNN training #####
        for (train_loss, train_reg, train_reg_sparse), setting_name, folder_path in zip(loss_settings, loss_settings_name, folder_path_list):
            print(f"loss setting: {setting_name}")
            print(f"Saved folder: {args.target}/{folder_path}")
            model = GNN(
                num_node_features,
                num_edge_features,
                hidden_dim=args.hidden_dim,
                mask_dim=args.mask_dim,
                num_layers=args.num_layers,
                conv_name=args.conv,
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
            epoch_list = []
            final_epoch = 0

            for epoch in range(1, args.epoch + 1):
                final_epoch += 1
                loss = train_epoch(
                    train_dataloader,
                    model,
                    optimizer,
                    loss_type=train_loss,
                    lambda1=args.lambda1,
                    lambda_group = args.lambda_group,
                    lambda_MSE = args.lambda_MSE,
                    regularization = train_reg,
                    Sparse = train_reg_sparse,
                )

                rmse_val, pcc_val = test_epoch(val_loader, model)

                # drive LR by validation performance
                scheduler.step(rmse_val)
                current_lr = scheduler.optimizer.param_groups[0]["lr"]

                rmse_test, pcc_test = test_epoch(test_dataloader, model)
                
                # Early stopping
                if epoch > 200:
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
                epoch_list.append(epoch)
                
            final_test_rmse = rmse_test
            final_test_pcc = pcc_test
            best_rmse_test = min(rmse_test_list)
            best_pcc_test = max(pcc_test_list)
            best_epoch_rmse_test = epoch_list[rmse_test_list.index(best_rmse_test)]
            best_epoch_pcc_test = epoch_list[pcc_test_list.index(best_pcc_test)]

            loss_metrics[setting_name].append({
                  'final_rmse': final_test_rmse,
                  'final_pcc': final_test_pcc,
                  'best_rmse': best_rmse_test,
                  'best_pcc': best_pcc_test
                })

            ###### Save the model performance metrics on th test data
                
            os.makedirs("cross_validation", exist_ok=True)
            os.makedirs(osp.join("cross_validation", args.cv_path), exist_ok=True)
            os.makedirs(osp.join("cross_validation", args.cv_path, args.target), exist_ok=True)
            os.makedirs(osp.join("cross_validation", args.cv_path, args.target, args.conv), exist_ok=True)
            os.makedirs(osp.join("cross_validation", args.cv_path, args.target, args.conv, folder_path), exist_ok=True)
            model_metrics_path = osp.join("cross_validation",
                args.cv_path, args.target, args.conv, folder_path, f"model_test_metrics_gnn_fold_{fold}_{args.conv}_{setting_name}_pool_{args.pool}_hiddenDim_{args.hidden_dim}_finalEpoch_{final_epoch}.csv"
            )
            res_dict = {
                "Epoch": epoch_list, 
                "Final test rmse": rmse_test_list,
                "Final test pcc": pcc_test_list,
                "Best epoch for rmse test": best_epoch_rmse_test,
                "Best test rmse": best_rmse_test,
                "Best epoch for pcc test": best_epoch_pcc_test,
                "Best test pcc": best_pcc_test,
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
    out_csv = f"cross_validation/{args.target}/{args.conv}/loss_cv_performance_topEpoch_{args.epoch}.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nSaved cross-validation summary to {out_csv}\n")
    

if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()
    args.loss_setting_list = ["MSE", "MSE+UCN", "MSE+N", "MSE+AC"]
    main_cv(args)
