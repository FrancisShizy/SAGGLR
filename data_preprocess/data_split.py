'''
How to run functions in data_split.py: 
1. Output data spliting into training and testing sets into ./data_preprocess/{kinase}
2. Output scaffold analysis reulst:

Under the folder of 
./SAGGLR/ 
        data_preprocess/
        SAGGLR/
(1) For single-time data splitting, run the code /SAGGLR/data_preprocess/data_split.py:

python -m data_preprocess.data_split --seed 1337 --data_path data_preprocess/data --test_set_fraction 0.2

Output folder: /SAGGLR/data_preprocess/scaffold_analysis/{kinase}/

(2) For cross-validation data splitting, run the code /SAGGLR/SAGGLR/cross_valid.py:

python -m SAGGLR.cross_valid_split

Output folder: /SAGGLR/cross_validation/{kinase}/scaffold_analysis/fold_{n}/
'''

import ast
import os
import os.path as osp
from collections.abc import Sequence
from typing import List, Tuple, Union

import dill
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import HeteroData

from data_preprocess.data_parser_utils import data_parser
from data_preprocess.featurization import create_pytorch_geometric_data_list_from_smiles_and_labels
from data_preprocess.scaffold_analysis import analyze_from_pairs_list, leakage_from_split_pairs


IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def create_heterodata(input: dict) -> HeteroData:
    """Converts a pair of molecules with smiles, masks, activities, and mcs to a HeteroData object."""
    hetero_data = HeteroData()

    data_i = create_pytorch_geometric_data_list_from_smiles_and_labels(
        input["smiles_i"], input["a_i"]
    )
    data_i.mask = torch.Tensor(input["mask_i"])
    data_i.smiles = input["smiles_i"]
    hetero_data["data_i"] = data_i

    data_j = create_pytorch_geometric_data_list_from_smiles_and_labels(
        input["smiles_j"], input["a_j"]
    )
    data_j.mask = torch.Tensor(input["mask_j"])
    data_j.smiles = input["smiles_j"]
    hetero_data["data_j"] = data_j

    hetero_data["mcs"] = input["mcs"]
    return hetero_data


def rebalance_pairs(
    train_pairs: List[HeteroData], test_pairs: List[HeteroData], test_set_fraction=0.2
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """Rebalances the pairs in the training and test sets to a (1-test)/(test) ratio."""
    n = len(train_pairs) / (1 - test_set_fraction)
    if len(test_pairs) > test_set_fraction * n:
        test_pairs = test_pairs[: int(test_set_fraction * n)]
    else:
        n = len(test_pairs) / test_set_fraction
        train_pairs = train_pairs[: int(n * (1 - test_set_fraction))]
    return train_pairs, test_pairs


def create_ligands_list(pairs_list: List[HeteroData]) -> List[str]:
    """Creates a unique list of ligands present in the pairs."""
    ligands_list = []
    for pair in pairs_list:
        ligands_list.append(pair["data_i"].smiles)
        ligands_list.append(pair["data_j"].smiles)
    return list(np.unique(ligands_list))


# ==========================
# (A) Scaffold-analysis split
# ==========================

def split_ligands_train_val_test(
    ligands_list: List[str],
    test_set_fraction: float,
    val_set_fraction: float,
    seed: int,
):
    """
    Split ligands into train/val/test.
    Note: val_set_fraction is fraction of *trainval pool*, not total.
    """
    trainval_lig, test_lig = train_test_split(
        ligands_list, test_size=test_set_fraction, random_state=seed, shuffle=True
    )
    train_lig, val_lig = train_test_split(
        trainval_lig, test_size=val_set_fraction, random_state=seed, shuffle=True
    )
    return list(train_lig), list(val_lig), list(test_lig)


def split_pairs_by_ligand_membership(
    pairs_list: List[HeteroData],
    train_ligands: set,
    val_ligands: set,
    test_ligands: set,
):
    """
    Only keep pairs where BOTH ligands are in the same split.
    (Pairs spanning splits are discarded.)
    """
    train_pairs, val_pairs, test_pairs = [], [], []
    for pair in pairs_list:
        si = pair["data_i"].smiles
        sj = pair["data_j"].smiles

        if (si in train_ligands) and (sj in train_ligands):
            train_pairs.append(pair)
        elif (si in val_ligands) and (sj in val_ligands):
            val_pairs.append(pair)
        elif (si in test_ligands) and (sj in test_ligands):
            test_pairs.append(pair)

    return train_pairs, val_pairs, test_pairs


def pairs_to_df(pairs: List[HeteroData]) -> pd.DataFrame:
    rows = []
    for p in pairs:
        rows.append((p["data_i"].smiles, p["data_j"].smiles))
    return pd.DataFrame(rows, columns=["smiles_i", "smiles_j"])


# =======================
# (B) Train/Test-only split
# =======================

def train_test_split_pairs(
    pairs_list: List[HeteroData],
    ligands_list: List[str],
    test_set_fraction: float,
    seed=42,
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """
    Split ligands into train/test and keep only pairs where BOTH ligands are in the same split.
    Then rebalance pairs to match the desired test fraction.
    """
    train_ligands, test_ligands = train_test_split(
        ligands_list, random_state=seed, test_size=test_set_fraction, shuffle=True
    )

    train_pairs, test_pairs = [], []
    train_ligands = set(train_ligands)
    test_ligands = set(test_ligands)

    for pair in pairs_list:
        si = pair["data_i"].smiles
        sj = pair["data_j"].smiles
        if (si in train_ligands) and (sj in train_ligands):
            train_pairs.append(pair)
        elif (si in test_ligands) and (sj in test_ligands):
            test_pairs.append(pair)

    train_pairs, test_pairs = rebalance_pairs(train_pairs, test_pairs, test_set_fraction)
    return train_pairs, test_pairs


# =======================
# Main
# =======================

if __name__ == "__main__":

    # where the raw processed pair-lines are stored
    data_process_path = "./data_preprocess/data/processed_data"

    args = data_parser().parse_args()

    # args.data_path is where you want kinase folders with train/test .pt
    os.makedirs(args.data_path, exist_ok=True)

    # scaffold analysis root folder
    os.makedirs(args.scaffold_analysis_root, exist_ok=True)

    for folder in os.listdir(data_process_path):
        print(folder)
        target = folder[:8]  # your kinase ID convention

        # ==========
        # Load pairs
        # ==========
        pairs_list = []
        with open(osp.join(data_process_path, folder), "r") as f:
            for line in f:
                inp = ast.literal_eval(line)
                pairs_list.append(create_heterodata(inp))

        ligands_list = create_ligands_list(pairs_list)

        # =========================================================
        # (1) DATA SPLIT OUTPUT (train/test ONLY) -> args.data_path
        #     Use train_test_split_pairs()
        # =========================================================
        dir_target = osp.join(args.data_path, target)
        os.makedirs(dir_target, exist_ok=True)

        train_pairs_tt, test_pairs_tt = train_test_split_pairs(
            pairs_list=pairs_list,
            ligands_list=ligands_list,
            test_set_fraction=args.test_set_fraction,
            seed=args.seed,
        )

        # Save train/test only (for model training)
        if len(train_pairs_tt) > 0 and len(test_pairs_tt) > 0:
            with open(osp.join(dir_target, f"{target}_seed_{args.seed}_train.pt"), "wb") as handle:
                dill.dump(train_pairs_tt, handle)
            with open(osp.join(dir_target, f"{target}_seed_{args.seed}_test.pt"), "wb") as handle:
                dill.dump(test_pairs_tt, handle)

            df_stats = pd.DataFrame(
                {
                    "target": [target],
                    "seed": [args.seed],
                    "n_compounds_total": [len(np.unique(ligands_list))],
                    "n_pairs_total": [len(pairs_list)],
                    "n_pairs_train": [len(train_pairs_tt)],
                    "n_pairs_test": [len(test_pairs_tt)],
                    "test_set_fraction": [args.test_set_fraction],
                }
            )
            df_stats.to_csv(osp.join(dir_target, f"{target}_seed_{args.seed}_stats.csv"), index=False)

        # =========================================================
        # (2) SCAFFOLD ANALYSIS OUTPUT (train/val/test) -> scaffold root
        #     Use split_ligands_train_val_test() + split_pairs_by_ligand_membership()
        # =========================================================
        out_sa = osp.join(args.scaffold_analysis_root, target)
        os.makedirs(out_sa, exist_ok=True)

        train_lig, val_lig, test_lig = split_ligands_train_val_test(
            ligands_list=ligands_list,
            test_set_fraction=args.test_set_fraction,
            val_set_fraction=args.val_set_fraction,
            seed=args.seed,
        )

        train_pairs_sa, val_pairs_sa, test_pairs_sa = split_pairs_by_ligand_membership(
            pairs_list=pairs_list,
            train_ligands=set(train_lig),
            val_ligands=set(val_lig),
            test_ligands=set(test_lig),
        )

        # Optional: rebalance train/test for scaffold split too (usually not necessary for analysis)
        # train_pairs_sa, test_pairs_sa = rebalance_pairs(train_pairs_sa, test_pairs_sa, args.test_set_fraction)

        # Save scaffold splits into scaffold folder
        with open(osp.join(out_sa, f"{target}_seed_{args.seed}_train.pt"), "wb") as handle:
            dill.dump(train_pairs_sa, handle)
        with open(osp.join(out_sa, f"{target}_seed_{args.seed}_val.pt"), "wb") as handle:
            dill.dump(val_pairs_sa, handle)
        with open(osp.join(out_sa, f"{target}_seed_{args.seed}_test.pt"), "wb") as handle:
            dill.dump(test_pairs_sa, handle)

        # Also save smiles tables for scaffold analysis + leakage report
        pairs_to_df(train_pairs_sa).to_csv(osp.join(out_sa, "train_pairs.csv"), index=False)
        pairs_to_df(val_pairs_sa).to_csv(osp.join(out_sa, "val_pairs.csv"), index=False)
        pairs_to_df(test_pairs_sa).to_csv(osp.join(out_sa, "test_pairs.csv"), index=False)

        # analysis on full + each split
        analyze_from_pairs_list(pairs_list, out_sa, prefix="all_")
        analyze_from_pairs_list(train_pairs_sa, out_sa, prefix="train_")
        analyze_from_pairs_list(val_pairs_sa, out_sa, prefix="val_")
        analyze_from_pairs_list(test_pairs_sa, out_sa, prefix="test_")

        leakage_from_split_pairs(
            train_pairs_df=pairs_to_df(train_pairs_sa),
            val_pairs_df=pairs_to_df(val_pairs_sa),
            test_pairs_df=pairs_to_df(test_pairs_sa),
            out_dir=out_sa,
            prefix="",
        )

        # Summary stats for scaffold split
        df_sa_stats = pd.DataFrame(
            {
                "target": [target],
                "seed": [args.seed],
                "n_pairs_total": [len(pairs_list)],
                "n_pairs_train_scaffold": [len(train_pairs_sa)],
                "n_pairs_val_scaffold": [len(val_pairs_sa)],
                "n_pairs_test_scaffold": [len(test_pairs_sa)],
                "n_lig_train": [len(train_lig)],
                "n_lig_val": [len(val_lig)],
                "n_lig_test": [len(test_lig)],
                "test_set_fraction": [args.test_set_fraction],
                "val_set_fraction": [args.val_set_fraction],
            }
        )
        df_sa_stats.to_csv(osp.join(out_sa, f"{target}_seed_{args.seed}_scaffold_split_stats.csv"), index=False)