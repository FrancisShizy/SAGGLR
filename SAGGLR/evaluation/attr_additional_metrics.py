# SAGGLR/evaluation/attr_additional_metrics.py

import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# =========================
# Constants / configuration
# =========================

MCS_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
EPS = 1e-12


# =========================
# Safe aggregation helpers
# =========================

def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _safe_nanmean(x: Sequence[float]) -> float:
    arr = _finite(np.asarray(x, dtype=float))
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _safe_nanstd(x: Sequence[float]) -> float:
    arr = _finite(np.asarray(x, dtype=float))
    return float(np.std(arr, ddof=0)) if arr.size > 0 else float("nan")


# =========================
# Core metrics (UPDATED)
# =========================

def _mol_metrics(mask: np.ndarray, attr: np.ndarray) -> Tuple[float, float, float, Dict[str, int]]:
    """
    UPDATED to match your dataset encoding:

      mask values:
        0  = atoms in common substructure (MCS scaffold)
        +1 = uncommon part of higher-activity compound
        -1 = uncommon part of lower-activity compound

    Key fix:
      - AUROC is NOT computed only on noncommon atoms (that often becomes single-class).
      - AUROC is computed against scaffold (0) as the opposite class:
          AUROC-pos: positives are mask==+1, negatives are mask!=+1 (includes 0 and -1)
          AUROC-neg: positives are mask==-1, negatives are mask!=-1 (includes 0 and +1),
                     using score = -attr (so "more negative attribution" -> higher score)
      - Spearman computed between attr and ternary mask {-1,0,+1} using all atoms.

    Returns:
      spearman, auroc_pos, auroc_neg, counts dict
    """
    mask = np.asarray(mask, dtype=float).reshape(-1)
    attr = np.asarray(attr, dtype=float).reshape(-1)

    keep = np.isfinite(mask) & np.isfinite(attr)
    mask = mask[keep]
    attr = attr[keep]

    n_all = int(mask.size)
    n_pos = int(np.sum(mask == 1))
    n_neg = int(np.sum(mask == -1))
    n_zero = int(np.sum(mask == 0))

    # Spearman(attr, mask) on {-1,0,+1}
    sp = float("nan")
    if n_all >= 2 and np.nanstd(mask) > EPS and np.nanstd(attr) > EPS:
        sp_val, _ = spearmanr(attr, mask)
        sp = float(sp_val)

    # AUROC (+1 vs others)
    auroc_pos = float("nan")
    y_pos = (mask == 1).astype(int)
    if np.unique(y_pos).size == 2:
        auroc_pos = float(roc_auc_score(y_pos, attr))

    # AUROC (-1 vs others), with score = -attr
    auroc_neg = float("nan")
    y_neg = (mask == -1).astype(int)
    if np.unique(y_neg).size == 2:
        auroc_neg = float(roc_auc_score(y_neg, -attr))

    return sp, auroc_pos, auroc_neg, {
        "n_all": n_all,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
    }


def _pair_metrics(hetero: HeteroData, colors_k: Tuple[Sequence[float], Sequence[float]]) -> Dict[str, object]:
    """
    Compute per-pair metrics as the mean over (data_i, data_j).
    Ground-truth comes from masks: data_i.mask, data_j.mask
    Attribution comes from colors list: (attr_i, attr_j)
    """
    data_i = hetero["data_i"]
    data_j = hetero["data_j"]
    attr_i, attr_j = colors_k

    mask_i = np.asarray(data_i.mask.detach().cpu().numpy(), dtype=float)
    mask_j = np.asarray(data_j.mask.detach().cpu().numpy(), dtype=float)

    attr_i = np.asarray(attr_i, dtype=float).reshape(-1)
    attr_j = np.asarray(attr_j, dtype=float).reshape(-1)

    sp_i, aucp_i, aucn_i, c_i = _mol_metrics(mask_i, attr_i)
    sp_j, aucp_j, aucn_j, c_j = _mol_metrics(mask_j, attr_j)

    # pairwise aggregate (nanmean so if one side undefined it can still use the other)
    sp_pair = _safe_nanmean([sp_i, sp_j])
    auc_pos_pair = _safe_nanmean([aucp_i, aucp_j])
    auc_neg_pair = _safe_nanmean([aucn_i, aucn_j])

    # MCS flags for thresholds
    mcs = np.asarray(hetero["mcs"], dtype=float).reshape(-1)
    # convert to 0/1 flags aligned with MCS_THRESHOLDS length
    mcs_flags = (mcs > 0).astype(int).tolist()
    if len(mcs_flags) != len(MCS_THRESHOLDS):
        # be robust if someone stored differently
        mcs_flags = (mcs_flags + [0] * len(MCS_THRESHOLDS))[: len(MCS_THRESHOLDS)]

    out = {
        "spearman": sp_pair,
        "auroc_pos": auc_pos_pair,
        "auroc_neg": auc_neg_pair,
        "n_all_i": c_i["n_all"],
        "n_all_j": c_j["n_all"],
        "n_zero_i": c_i["n_zero"],
        "n_zero_j": c_j["n_zero"],
        "n_pos_i": c_i["n_pos"],
        "n_pos_j": c_j["n_pos"],
        "n_neg_i": c_i["n_neg"],
        "n_neg_j": c_j["n_neg"],
    }

    # add mcs_* columns
    for thr, flag in zip(MCS_THRESHOLDS, mcs_flags):
        out[f"mcs_{thr}"] = int(flag)

    return out


# =========================
# Plot helpers
# =========================

def _plot_metric_vs_threshold(df: pd.DataFrame, metric_col: str, title: str, out_path: str):
    """
    df is the threshold summary table with columns:
      mcs_threshold, {metric_col}_mean, {metric_col}_std
    """
    if df.empty:
        return

    x = df["mcs_threshold"].to_numpy()
    y = df[f"{metric_col}_mean"].to_numpy(dtype=float)
    yerr = df[f"{metric_col}_std"].to_numpy(dtype=float)

    # If everything is NaN, don't create an empty plot.
    if not np.isfinite(y).any():
        return

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3)
    plt.xlabel("MCS threshold")
    plt.ylabel(metric_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_metric_vs_drop(drop_list: Sequence[float], y_list: Sequence[float], title: str, out_path: str):
    drop = np.asarray(drop_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    if y.size == 0 or (not np.isfinite(y).any()):
        return

    plt.figure()
    plt.plot(drop, y, marker="o")
    plt.xlabel("Edge drop probability")
    plt.ylabel("Metric")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================
# Public API: additional metrics
# =========================

def run_and_save_additional_metrics(
    pairs_list: List[HeteroData],
    colors: List[Tuple[Sequence[float], Sequence[float]]],
    out_dir: str,
    prefix: str,
) -> Dict[str, object]:
    """
    Produces:
      1) {prefix}_pairwise_additional_metrics.csv
      2) {prefix}_additional_metrics_summary_by_mcs.csv
      3) plots for Spearman/AUROC vs MCS threshold

    IMPORTANT:
      - Uses masks like get_scores(): data_i.mask, data_j.mask
      - Uses attributions like get_colors(): colors[k] = [attr_i, attr_j]
      - AUROC is defined even when masks are {0,-1} or {0,+1}
    """
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for k, hetero in enumerate(pairs_list):
        row = _pair_metrics(hetero, colors[k])
        row["pair_idx"] = k
        rows.append(row)

    per_pair_df = pd.DataFrame(rows)

    # save per-pair table
    per_pair_path = osp.join(out_dir, f"{prefix}_pairwise_additional_metrics.csv")
    per_pair_df.to_csv(per_pair_path, index=False)

    # Build threshold summary
    summary_rows = []
    for thr in MCS_THRESHOLDS:
        flag_col = f"mcs_{thr}"
        if flag_col not in per_pair_df.columns:
            continue
        sub = per_pair_df[per_pair_df[flag_col] == 1]

        summary_rows.append({
            "mcs_threshold": thr,

            "spearman_mean": _safe_nanmean(sub["spearman"]) if len(sub) > 0 else float("nan"),
            "spearman_std": _safe_nanstd(sub["spearman"]) if len(sub) > 0 else float("nan"),

            "auroc_pos_mean": _safe_nanmean(sub["auroc_pos"]) if len(sub) > 0 else float("nan"),
            "auroc_pos_std": _safe_nanstd(sub["auroc_pos"]) if len(sub) > 0 else float("nan"),

            "auroc_neg_mean": _safe_nanmean(sub["auroc_neg"]) if len(sub) > 0 else float("nan"),
            "auroc_neg_std": _safe_nanstd(sub["auroc_neg"]) if len(sub) > 0 else float("nan"),

            "n_pairs": int(len(sub)),
        })

    summary_df = pd.DataFrame(summary_rows)

    # add overall row (optional but helpful)
    overall = {
        "mcs_threshold": "overall",
        "spearman_mean": _safe_nanmean(per_pair_df["spearman"]) if len(per_pair_df) > 0 else float("nan"),
        "spearman_std": _safe_nanstd(per_pair_df["spearman"]) if len(per_pair_df) > 0 else float("nan"),
        "auroc_pos_mean": _safe_nanmean(per_pair_df["auroc_pos"]) if len(per_pair_df) > 0 else float("nan"),
        "auroc_pos_std": _safe_nanstd(per_pair_df["auroc_pos"]) if len(per_pair_df) > 0 else float("nan"),
        "auroc_neg_mean": _safe_nanmean(per_pair_df["auroc_neg"]) if len(per_pair_df) > 0 else float("nan"),
        "auroc_neg_std": _safe_nanstd(per_pair_df["auroc_neg"]) if len(per_pair_df) > 0 else float("nan"),
        "n_pairs": int(len(per_pair_df)),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    summary_path = osp.join(out_dir, f"{prefix}_additional_metrics_summary_by_mcs.csv")
    summary_df.to_csv(summary_path, index=False)

    # plots (only over numeric thresholds)
    numeric_summary = summary_df[summary_df["mcs_threshold"].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))]
    if len(numeric_summary) > 0:
        _plot_metric_vs_threshold(
            numeric_summary, "spearman",
            "Spearman(attr, mask{-1,0,+1}) vs MCS threshold",
            osp.join(out_dir, f"{prefix}_spearman_vs_mcs.png"),
        )
        _plot_metric_vs_threshold(
            numeric_summary, "auroc_pos",
            "AUROC(+1 vs others, score=attr) vs MCS threshold",
            osp.join(out_dir, f"{prefix}_auroc_pos_vs_mcs.png"),
        )
        _plot_metric_vs_threshold(
            numeric_summary, "auroc_neg",
            "AUROC(-1 vs others, score=-attr) vs MCS threshold",
            osp.join(out_dir, f"{prefix}_auroc_neg_vs_mcs.png"),
        )

    return {
        "per_pair_path": per_pair_path,
        "summary_path": summary_path,
    }


# =========================
# Perturbation stability (edge-drop)
# =========================

def _clone_data_with_batch_zero(d: Data) -> Data:
    # shallow clone basic fields; preserve mask/y/smiles if present
    out = Data(
        x=d.x,
        edge_index=d.edge_index,
        edge_attr=getattr(d, "edge_attr", None),
        y=getattr(d, "y", None),
        mask=getattr(d, "mask", None),
        smiles=getattr(d, "smiles", None),
    )
    out.batch = torch.zeros(out.x.size(0), dtype=torch.long, device=out.x.device)
    return out


def _edge_drop_data(d: Data, drop_prob: float, rng: np.random.RandomState) -> Data:
    """
    Drops edges independently. Works with directed edge_index (your graphs store both directions).
    """
    if drop_prob <= 0:
        return _clone_data_with_batch_zero(d)

    edge_index = d.edge_index
    E = edge_index.size(1)
    if E == 0:
        return _clone_data_with_batch_zero(d)

    keep = rng.rand(E) >= drop_prob
    keep_t = torch.from_numpy(keep).to(edge_index.device)

    new_edge_index = edge_index[:, keep_t]
    new_edge_attr = None
    if getattr(d, "edge_attr", None) is not None:
        new_edge_attr = d.edge_attr[keep_t]

    out = Data(
        x=d.x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=getattr(d, "y", None),
        mask=getattr(d, "mask", None),
        smiles=getattr(d, "smiles", None),
    )
    out.batch = torch.zeros(out.x.size(0), dtype=torch.long, device=out.x.device)
    return out


def run_and_save_perturbation_stability(
    pairs_list: List[HeteroData],
    explainer,
    out_dir: str,
    prefix: str,
    drop_list: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3),
    n_repeats: int = 3,
    base_seed: int = 0,
) -> Dict[str, object]:
    """
    For each edge-drop probability:
      - perturb each molecule's graph
      - recompute attribution with explainer.explain_graph(data)
      - compute per-pair "spearman" using UPDATED ternary mask metric (attr vs {-1,0,+1})
      - aggregate over pairs and repeats

    Outputs:
      - {prefix}_perturbation_stability_summary.csv
      - plot: {prefix}_perturb_spearman.png
    """
    os.makedirs(out_dir, exist_ok=True)

    drop_list = list(drop_list)
    spearman_rep = []  # list of lists: repeats x drop

    for r in range(n_repeats):
        rng = np.random.RandomState(base_seed + 1000 * r)
        spearman_this_repeat = []

        for drop in drop_list:
            sp_pairs = []

            for hetero in pairs_list:
                d_i: Data = hetero["data_i"]
                d_j: Data = hetero["data_j"]

                d_i_p = _edge_drop_data(d_i, drop, rng)
                d_j_p = _edge_drop_data(d_j, drop, rng)

                # explainer expects .batch as in get_colors()
                # (we set batch=0 inside _edge_drop_data)
                attr_i = explainer.explain_graph(d_i_p)
                attr_j = explainer.explain_graph(d_j_p)

                mask_i = np.asarray(d_i.mask.detach().cpu().numpy(), dtype=float)
                mask_j = np.asarray(d_j.mask.detach().cpu().numpy(), dtype=float)

                sp_i, _, _, _ = _mol_metrics(mask_i, np.asarray(attr_i, dtype=float))
                sp_j, _, _, _ = _mol_metrics(mask_j, np.asarray(attr_j, dtype=float))

                sp_pairs.append(_safe_nanmean([sp_i, sp_j]))

            spearman_this_repeat.append(_safe_nanmean(sp_pairs))

        spearman_rep.append(spearman_this_repeat)

    spearman_rep = np.asarray(spearman_rep, dtype=float)  # shape (n_repeats, n_drop)
    spearman_mean = np.nanmean(spearman_rep, axis=0).tolist()
    spearman_std = np.nanstd(spearman_rep, axis=0).tolist()

    summary = pd.DataFrame({
        "drop": drop_list,
        "spearman_mean": spearman_mean,
        "spearman_std": spearman_std,
    })

    out_csv = osp.join(out_dir, f"{prefix}_perturbation_stability_summary.csv")
    summary.to_csv(out_csv, index=False)

    _plot_metric_vs_drop(
        drop_list,
        summary["spearman_mean"].tolist(),
        "Spearman(attr, mask{-1,0,+1}) under perturbation",
        osp.join(out_dir, f"{prefix}_perturb_spearman.png"),
    )

    return {"summary_csv": out_csv}