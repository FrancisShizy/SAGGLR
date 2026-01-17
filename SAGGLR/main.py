'''
This main.py file realize these functions below:
1. load preprocessed activity-cliff pair traing & testing datasets from data+preprocess/data/{kinase} using DataLoader.
    Please first run ./data_preprocess/data_split.py to create train/test sets
    
2. Train GNN models & save model performance under various combination of loss settings and model settings:
    (1) Loss_settings (basic loss + regularization):                                                           
        a. MSE+UCN without group lasso                        
        b. MSE+N without group lasso 
        c. MSE+N with group lasso
        d. MSE+N with sparse group lasso) 
    (2) Model settings (GNN layers + Pooling layers): 
        a. NNCov + Mean
        b. GAT + Mean
        c. GIN + Add
3. Evaluate feature attribution performance via:
    (1) Calculate accuarcy & F1 scores for local atom-level color labels (get_scores).
    (2) Estimate global direction score (get_global_directions). 
    (3) Additional interpretability metrics (Spearman + AUROC) + stability (e.g., sensitivity to graph perturbation).

How to run:
bash main.sh {cam|gradcam|gradinput|ig}

Saving pathway:
    # Model Performance: SAGGLR/logs/{kinase}}/{conv_pooling_settings}/{loss_settings}/.
    # Model paramters: SAGGLR/logs/{kinase}}/{conv_pooling,settings}/{loss_settings}/.
    # Color masks: SAGGLR/colors/feature_attribution/{kinase}}/{conv_pooling,settings}/{loss_settings}/.
    # Feature attribution performance: SAGGLE/feature_attribution/{kinase}}/{conv_pooling,settings}/{loss_settings}/.

'''
import os
import os.path as osp
import time
import sys
import dill
import numpy as np
import pandas as pd
import torch
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
import matplotlib.pyplot as plt

from SAGGLR.evaluation.attr_additional_metrics import (
    run_and_save_additional_metrics,
    run_and_save_perturbation_stability,
)


def main(args):
    set_seed(args.seed)
    torch.cuda.empty_cache()

    conv_pooling_settings = (f"{args.conv_main}_{args.pool_main}")
    print(f"Conv & pooling layers used in training process: {conv_pooling_settings}")
    print(f"Basic loss: {args.loss}")

    if args.loss == "MSE+N":
        if args.regularization:
            if args.Sparse: # MSE+N+SGL
                w_penalty_or_not = "w_sparse_group_lasso"
                loss_settings = "MSE_N_SGL"
                model_loss_settings = f"{conv_pooling_settings}_{loss_settings}"
                train_params = (
                    f"{model_loss_settings}_{args.explainer}_hiddenDim_{args.hidden_dim_gnn}_lambda_{args.lambda_group}"
                )
            else: # MSE+N+GL
                w_penalty_or_not = "w_group_lasso"
                loss_settings = "MSE_N_GL"
                model_loss_settings = f"{conv_pooling_settings}_{loss_settings}"
                train_params = (
                    f"{model_loss_settings}_{args.explainer}_hiddenDim_{args.hidden_dim_gnn}_lambda_{args.lambda_group}"
                )
        else: # MSE+N
            w_penalty_or_not = "wo_lasso"
            loss_settings = "MSE_N"
            model_loss_settings = f"{conv_pooling_settings}_{loss_settings}"
            train_params = (
                    f"{model_loss_settings}_{args.explainer}_hiddenDim_{args.hidden_dim_gnn}"
                )
    elif args.loss == "MSE+UCN": # MSE+UCN
        w_penalty_or_not = "wo_lasso"
        loss_settings = "MSE_UCN"
        model_loss_settings = f"{conv_pooling_settings}_{loss_settings}"
        train_params = (
                f"{model_loss_settings}_{args.explainer}_hiddenDim_{args.hidden_dim_gnn}"
            )
    print(f"Train models with model & loss settings: {model_loss_settings}.")

    
    ############################################# 
    ##### Data loading and pre-processing #######
    ############################################# 
    print(f"Data loading path: {args.data_path}")

    # Check that data exists
    file_train = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_train.pt"
    )
    file_test = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_test.pt"
    )

    if not osp.exists(file_train) or not osp.exists(file_test):
        raise FileNotFoundError(
            "Data not found. Please try to choose another protein target"
        )
    
    with open(file_train, "rb") as handle:
        trainval_dataset = dill.load(handle)

    with open(file_test, "rb") as handle:
        test_dataset = dill.load(handle)

    # Ligands in testing set are NOT in training set!

    train_dataset, val_dataset = train_test_split(
        trainval_dataset, random_state=args.seed, test_size=args.val_set_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=args.num_workers,  # num_workers can go into args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    num_node_features, num_edge_features = get_num_features(train_dataset[0])


    ############################################# 
    ##### GNN training ##########################
    ############################################# 
    model = GNN(
        num_node_features,
        num_edge_features,
        hidden_dim_embed = args.hidden_dim_embed, 
        hidden_dim_gnn = args.hidden_dim_gnn,
        hidden_dim_mlp = args.hidden_dim_mlp,
        hidden_dim_linear = args.hidden_dim_linear,
        mask_dim=args.mask_dim,
        num_layers=args.num_layers,
        conv_name=args.conv_main,
        pool=args.pool_main,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_main)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4
    )
    min_error = None

    # Early stopping
    last_loss = 100
    patience = 5
    trigger_times = 0
    rmse_test_list = []
    pcc_test_list = []
    epoch_list = []

    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_type=args.loss,
            lambda1=args.lambda1,
            lambda_group = args.lambda_group,
            lambda_MSE = args.lambda_MSE,
            regularization = args.regularization,
            Sparse = args.Sparse,
        )
        
        # Adjust the learning rate based on validation performance
        rmse_val, pcc_val = test_epoch(val_loader, model)
        scheduler.step(rmse_val)
        if min_error is None or rmse_val <= min_error:
            min_error = rmse_val

        t2 = time.time()
        rmse_test, pcc_test = test_epoch(test_loader, model)
        t3 = time.time()
        
        rmse_test_list.append(rmse_test)
        pcc_test_list.append(pcc_test)
        epoch_list.append(epoch)

        if epoch % args.verbose == 10:
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, Test PCC: {:.5f}".format(
                    epoch, t3 - t1, lr, loss, rmse_test, pcc_test
                )
            )
            continue

        print(
            "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Validation Loss: {:.5f}, Val PCC: {:.5f}".format(
                epoch, t2 - t1, lr, loss, rmse_val, pcc_val
            )
        )

        # Early stopping
        if epoch > 50:
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
        
    print("Final test rmse: {:.10f}".format(rmse_test))
    print("Final test pcc: {:.10f}".format(pcc_test))
    best_rmse_test = min(rmse_test_list)
    best_pcc_test = max(pcc_test_list)
    best_epoch_rmse_test = epoch_list[rmse_test_list.index(best_rmse_test)]
    best_epoch_pcc_test = epoch_list[pcc_test_list.index(best_pcc_test)]

    # Save GNN scores
    os.makedirs(osp.join(args.log_path, args.target, conv_pooling_settings, loss_settings), exist_ok=True)
    model_res_path = osp.join(
        args.log_path, args.target, conv_pooling_settings, loss_settings, f"model_scores_gnn_{train_params}.csv"
    )
    df = pd.DataFrame(
        [
            [
                args.target,
                args.seed,
                args.conv_main,
                args.pool_main,
                args.loss,
                args.lambda_group,
                args.explainer,
                loss_settings,
                rmse_test,
                pcc_test,
            ]
        ],
        columns=[
            "target",
            "seed",
            "conv",
            "pool",
            "loss",
            "lambda_group",
            "explainer",
            "penalty",
            "rmse_test",
            "pcc_test",
        ],
    )
    df.to_csv(model_res_path, index=False)
    
    ###### Save the model performance metrics on th test data
    model_metrics_path = osp.join(
        args.log_path, args.target, conv_pooling_settings, loss_settings, f"model_test_metrics_gnn_{train_params}.csv"
    )
    res_dict = {
        "Epoch": epoch_list, 
        "Test rmse": rmse_test_list,
        "Test pcc": pcc_test_list,
        "Best epoch for rmse test": best_epoch_rmse_test,
        "Best test rmse": best_rmse_test,
        "Best epoch for pcc test": best_epoch_pcc_test,
        "Best test pcc": best_pcc_test,
    }
    res_df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
    res_df.to_csv(model_metrics_path, index=False)
    print(f"Model performance metrics on th test data saved to: {model_metrics_path}")
    
    ##### Save the model
    model_path = osp.join(
        args.log_path, args.target, conv_pooling_settings, loss_settings, f"model_{train_params}.pth"
    )
    torch.save(model, model_path)
    

    ############################################# 
    ##### Feature Attribution #####
    #############################################
    model.eval()

    if args.explainer == "gradinput":
        explainer = GradInput(DEVICE, model)
    elif args.explainer == "ig":
        explainer = IntegratedGradient(DEVICE, model)
    elif args.explainer == "cam":
        explainer = CAM(DEVICE, model)
    elif args.explainer == "gradcam":
        explainer = GradCAM(DEVICE, model)


    t0 = time.time()
    train_colors = get_colors(trainval_dataset, explainer)
    time_explainer = (time.time() - t0) / len(trainval_dataset)
    print("Average time to generate 1 explanation: ", time_explainer)
    test_colors = get_colors(test_dataset, explainer)

    # Save colors
    os.makedirs(osp.join(args.color_path, args.explainer, args.target, conv_pooling_settings, loss_settings), exist_ok=True)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            conv_pooling_settings,
            loss_settings,
            f"{args.target}_seed_{args.seed}_{train_params}_train.pt",
        ), "wb",
    ) as handle:
        dill.dump(train_colors, handle)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            conv_pooling_settings,
            loss_settings,
            f"{args.target}_seed_{args.seed}_{train_params}_test.pt",
        ), "wb",
    ) as handle:
        dill.dump(test_colors, handle)

    accs_train, f1s_train = get_scores(trainval_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(
        trainval_dataset, train_colors, set="train"
    )
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    os.makedirs(osp.join(args.feature_attri_path, args.explainer, args.target, conv_pooling_settings, loss_settings), exist_ok=True)
    global_res_path = osp.join(
        args.feature_attri_path, args.explainer, args.target, conv_pooling_settings, loss_settings, f"attr_scores_{train_params}_{args.target}.csv"
    )

    accs_train, f1s_train = (
        np.nanmean(accs_train, axis=0).tolist(),
        np.nanmean(f1s_train, axis=0).tolist(),
    )
    accs_test, f1s_test = (
        np.nanmean(accs_test, axis=0).tolist(),
        np.nanmean(f1s_test, axis=0).tolist(),
    )
    global_dir_train, global_dir_test = (
        np.nanmean(global_dir_train, axis=0).tolist(),
        np.nanmean(global_dir_test, axis=0).tolist(),
    )
    n_mcs_train, n_mcs_test = (
        np.sum(get_mcs(trainval_dataset), axis=0).tolist(),
        np.sum(get_mcs(test_dataset), axis=0).tolist(),
    )

    res_dict = {
        "target": [args.target] * 10,
        "seed": [args.seed] * 10,
        "conv": [args.conv_main] * 10,
        "pool": [args.pool_main] * 10,
        "loss": [args.loss] * 10,
        "lambda1": [args.lambda1] * 10,
        "explainer": [args.explainer] * 10,
        "penalty": [w_penalty_or_not] * 10,
        "time": [round(time_explainer, 4)] * 10,
        "acc_train": accs_train,
        "acc_test": accs_test,
        "f1_train": f1s_train,
        "f1_test": f1s_test,
        "global_dir_train": global_dir_train,
        "global_dir_test": global_dir_test,
        "mcs": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "n_mcs_train": n_mcs_train,
        "n_mcs_test": n_mcs_test,
    }
    df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
    df.to_csv(global_res_path, index=False)
    print(f"metrics saved to: {global_res_path}")

    ################################################################################################################
    # Additional interpretability metrics (Spearman + AUROC) + stability (e.g., sensitivity to graph perturbation)
    ################################################################################################################
    extra_out_dir = osp.join(args.feature_attri_path, args.explainer, args.target, conv_pooling_settings, loss_settings, "additional_metrics")
    os.makedirs(extra_out_dir, exist_ok=True)

    # Train/val pooled set (trainval_dataset) + test set
    _ = run_and_save_additional_metrics(
        pairs_list=trainval_dataset,
        colors=train_colors,
        out_dir=extra_out_dir,
        prefix=f"{args.target}_seed_{args.seed}_{train_params}_train",
    )
    _ = run_and_save_additional_metrics(
        pairs_list=test_dataset,
        colors=test_colors,
        out_dir=extra_out_dir,
        prefix=f"{args.target}_seed_{args.seed}_{train_params}_test",
    )
    print(f"Spearman + AUROC results saved to: {extra_out_dir}")

    # ===== Perturbation stability (edge-drop) =====
    # Uses the same explainer object you already created (GradInput / IG / CAM / GradCAM)
    stability_out_dir = osp.join(extra_out_dir, "perturbation_stability")
    os.makedirs(stability_out_dir, exist_ok=True)

    _ = run_and_save_perturbation_stability(
        pairs_list=test_dataset,
        explainer=explainer,
        out_dir=stability_out_dir,
        prefix=f"{args.target}_seed_{args.seed}_{train_params}_test",
        drop_list=[0.0, 0.05, 0.1, 0.2, 0.3],
        n_repeats=3,
        base_seed=args.seed,
    )
    print(f"Perturbation stability results saved to: {stability_out_dir}")

if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()

    for loss in ["MSE+N", "MSE+UCN"]:
        args.loss = loss
        main(args)
