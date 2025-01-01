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
from torch_geometric.data import DataLoader

from SAGGLR.evaluation.explain_color import get_scores
from SAGGLR.evaluation.explain_direction_global import get_global_directions
from SAGGLR.evaluation.feature_attribution import CAM, GradCAM, GradInput, IntegratedGradient
from SAGGLR.gnn_framework.model import GNN
from SAGGLR.utils.parser_utils import overall_parser
from SAGGLR.utils.train_utils import DEVICE, test_epoch, train_epoch
from SAGGLR.utils.utils import get_num_features, get_colors, get_mcs, set_seed

def main(args):
    set_seed(args.seed)
    torch.cuda.empty_cache()
    
    if args.regularization:
        train_params = (
            f"{args.conv}_{args.loss}_{args.pool}_{args.explainer}_hiddenDim_{args.hidden_dim}_GL_{args.lambda_group}"
        )
    else:
        train_params = (
            f"{args.conv}_{args.loss}_{args.pool}_{args.explainer}_hiddenDim_{args.hidden_dim}"
        )

    ##### Data loading and pre-processing #####
    print(args.data_path)
    print(args.result_path)

    # Check that data exists
    file_train = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_train.pt"
    )
    file_test = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_test.pt"
    )

    if not osp.exists(file_train) or not osp.exists(file_test):
        raise FileNotFoundError(
            "Data not found. Please try to - choose another protein target or - run code/pair.py with a new seed."
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

    ##### GNN training #####
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
    rmse_test_output = []
    pcc_test_output = []
    epoch_output =[]

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

        if epoch % args.verbose == 0:
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, Test PCC: {:.5f}".format(
                    epoch, t3 - t1, lr, loss, rmse_test, pcc_test
                )
            )
            rmse_test_output.append(rmse_test)
            pcc_test_output.append(pcc_test)
            epoch_output.append(epoch)
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
    rmse_test_output.append(rmse_test)
    pcc_test_output.append(pcc_test)
    epoch_output.append(epoch)
    best_rmse_test = min(rmse_test_list)
    best_pcc_test = max(pcc_test_output)
    best_epoch_rmse_test = epoch_list[rmse_test_list.index(best_rmse_test)]
    best_epoch_pcc_test = epoch_list[pcc_test_list.index(best_pcc_test)]

    # Saving pathway:
    # Model Performance: myfolder/logs/protein_name/w_penalty/.
    # ModeL paramters: myfolder/logs/protein_name/w_penalty/.
    # Color masks: myfolder/colors/feature_attr/protein_name/w_penalty/.
    # Feature attribution performance: myfolder/feature_attr/protein_name/w_penalty/.

    # Save GNN scores
    if args.regularization:
        if args.Sparse:
            folder_path = "w_sparse_group_lasso"
        else:
            folder_path = "w_group_lasso"
    else:
        folder_path = "wo_lasso"
    print(folder_path)
        
    os.makedirs(osp.join(args.log_path, args.target, folder_path), exist_ok=True)
    global_res_path = osp.join(
        args.log_path, args.target, folder_path, f"model_scores_gnn_{train_params}.csv"
    )
    df = pd.DataFrame(
        [
            [
                args.target,
                args.seed,
                args.conv,
                args.pool,
                args.loss,
                args.lambda_group,
                args.explainer,
                folder_path,
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
    df.to_csv(global_res_path, index=False)
    
    ###### Save the model performance metrics on th test data
    model_metrics_path = osp.join(
        args.log_path, args.target, folder_path, f"model_test_metrics_gnn_{train_params}.csv"
    )
    res_dict = {
        "Epoch": epoch_output, 
        "Test rmse": rmse_test_output,
        "Test pcc": pcc_test_output,
        "Best epoch for rmse test": best_epoch_rmse_test,
        "Best test rmse": best_rmse_test,
        "Best epoch for pcc test": best_epoch_pcc_test,
        "Best test pcc": best_pcc_test,
    }
    df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
    df.to_csv(model_metrics_path, index=False)
    
    ##### Save the model
    model_path = osp.join(
        args.log_path, args.target, folder_path, f"model_{train_params}.pth"
    )
    torch.save(model, model_path)
    

    ##### Feature Attribution #####
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
    os.makedirs(osp.join(args.color_path, args.explainer, args.target, folder_path), exist_ok=True)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            folder_path,
            f"{args.target}_seed_{args.seed}_{train_params}_train.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(train_colors, handle)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            folder_path,
            f"{args.target}_seed_{args.seed}_{train_params}_test.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(test_colors, handle)

    accs_train, f1s_train = get_scores(trainval_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(
        trainval_dataset, train_colors, set="train"
    )
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    os.makedirs(osp.join(args.result_path, args.explainer, args.target, folder_path), exist_ok=True)
    global_res_path = osp.join(
        args.result_path, args.explainer, args.target, folder_path, f"attr_scores_{train_params}_{args.target}.csv"
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
        "conv": [args.conv] * 10,
        "pool": [args.pool] * 10,
        "loss": [args.loss] * 10,
        "lambda1": [args.lambda1] * 10,
        "explainer": [args.explainer] * 10,
        "penalty": [folder_path] * 10,
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
    torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()

    for loss in ["MSE+N", "MSE", "MSE+AC"]:
        args.loss = loss
        main(args)
