# data_preprocess/scaffold_analysis/scaffold_analysis.py
'''
Description:
(1) compute % pairs sharing identical cores
(2) detect overrepresented compounds (e.g., 50–100+ cliff pairs)
(3) compute scaffold frequency distributions
(4) check compound overlap (leakage) across train/val/test
save everything into:
	•	scaffold_analysis/{kinase}/... for one-time split
	•	cross_validation/{kinase}/scaffold_analysis/fold_{k}/... for CV
'''

from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Iterable, Optional

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# Silence RDKit warnings like: "non-ring atom marked aromatic"
RDLogger.DisableLog("rdApp.*")


def _safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES robustly without crashing the pipeline."""
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is not None:
            return mol
    except Exception:
        pass

    # Fallback: sanitize manually
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canon_smiles(smiles: str) -> Optional[str]:
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def murcko_scaffold_smiles(smiles: str, include_chirality: bool = False) -> Optional[str]:
    """Bemis–Murcko scaffold SMILES."""
    mol = _safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=include_chirality)
    except Exception:
        return None


def pair_core_key(smiles_i: str, smiles_j: str) -> Optional[str]:
    """
    A canonical 'core key' for a pair: sorted scaffold SMILES.
    Returns None if either scaffold cannot be computed.
    """
    si = murcko_scaffold_smiles(smiles_i)
    sj = murcko_scaffold_smiles(smiles_j)
    if si is None or sj is None:
        return None
    a, b = sorted([si, sj])
    return f"{a}||{b}"


@dataclass
class OverrepConfig:
    threshold_50: int = 50
    threshold_100: int = 100


@dataclass
class LeakageReport:
    # sets overlap counts
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    # overlap lists (small to moderate sizes; if huge, we also save to CSV)
    train_val_compounds: list[str]
    train_test_compounds: list[str]
    val_test_compounds: list[str]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _extract_compounds_from_pairs_df(df_pairs: pd.DataFrame) -> set[str]:
    return set(df_pairs["smiles_i"]).union(set(df_pairs["smiles_j"]))


def _pairs_df_from_pairs_list(pairs_list: Iterable[dict] | Iterable[object]) -> pd.DataFrame:
    """
    Accepts:
      - list of dict: {"smiles_i","smiles_j",...}
      - list of HeteroData: pair["data_i"].smiles, pair["data_j"].smiles
    """
    rows = []
    for p in pairs_list:
        if isinstance(p, dict):
            si = p.get("smiles_i", None)
            sj = p.get("smiles_j", None)
        else:
            # HeteroData
            si = getattr(p["data_i"], "smiles", None)
            sj = getattr(p["data_j"], "smiles", None)
        si = canon_smiles(si) if si is not None else None
        sj = canon_smiles(sj) if sj is not None else None
        if si is None or sj is None:
            continue
        rows.append((si, sj))
    return pd.DataFrame(rows, columns=["smiles_i", "smiles_j"])


def analyze_pairs(
    pairs_df: pd.DataFrame,
    out_dir: str,
    overrep_cfg: OverrepConfig = OverrepConfig(),
    prefix: str = "",
) -> dict:
    """
    pairs_df must contain columns: smiles_i, smiles_j (canonical recommended).
    Saves multiple CSV/JSON summary files into out_dir.
    Returns a dict summary.
    """
    ensure_dir(out_dir)

    df = pairs_df.copy()
    df = df.dropna(subset=["smiles_i", "smiles_j"]).reset_index(drop=True)

    # Canonicalize SMILES (important for consistent overlap checks)
    df["smiles_i"] = df["smiles_i"].map(lambda s: canon_smiles(s) or s)
    df["smiles_j"] = df["smiles_j"].map(lambda s: canon_smiles(s) or s)

    # --- Scaffold per compound ---
    all_smiles = pd.unique(pd.concat([df["smiles_i"], df["smiles_j"]], ignore_index=True))
    scaf_map = {}
    for s in all_smiles:
        scaf_map[s] = murcko_scaffold_smiles(s)

    # scaffold frequency (per compound)
    scafs = [v for v in scaf_map.values() if v is not None]
    scaffold_counts = Counter(scafs)
    df_scaffold_freq = (
        pd.DataFrame(scaffold_counts.items(), columns=["scaffold_smiles", "n_compounds"])
        .sort_values("n_compounds", ascending=False)
        .reset_index(drop=True)
    )
    df_scaffold_freq.to_csv(os.path.join(out_dir, f"{prefix}scaffold_frequency_compounds.csv"), index=False)

    # Map pair -> core key (scaffold_i||scaffold_j)
    core_keys = []
    ok_pair_mask = []
    for si, sj in zip(df["smiles_i"], df["smiles_j"]):
        ck = pair_core_key(si, sj)
        core_keys.append(ck)
        ok_pair_mask.append(ck is not None)
    df["pair_core_key"] = core_keys
    df_ok = df.loc[ok_pair_mask].reset_index(drop=True)

    # 1) proportion of pairs that share identical cores (i.e., core_key appears >1)
    core_count = Counter(df_ok["pair_core_key"].tolist())
    n_pairs_ok = len(df_ok)
    n_pairs_in_repeated_cores = sum(c for c in core_count.values() if c > 1)
    proportion_pairs_repeated_core = (n_pairs_in_repeated_cores / n_pairs_ok) if n_pairs_ok else 0.0

    df_core_freq = (
        pd.DataFrame(core_count.items(), columns=["pair_core_key", "n_pairs"])
        .sort_values("n_pairs", ascending=False)
        .reset_index(drop=True)
    )
    df_core_freq.to_csv(os.path.join(out_dir, f"{prefix}pair_core_frequency.csv"), index=False)

    # also save pairs with scaffolds (useful for debugging)
    df_ok["scaffold_i"] = df_ok["smiles_i"].map(scaf_map)
    df_ok["scaffold_j"] = df_ok["smiles_j"].map(scaf_map)
    df_ok.to_csv(os.path.join(out_dir, f"{prefix}pairs_with_scaffolds.csv"), index=False)

    # 2) Overrepresented compounds in many pairs
    # count each compound participation across all pairs
    compound_pair_counts = Counter()
    for si, sj in zip(df["smiles_i"], df["smiles_j"]):
        compound_pair_counts[si] += 1
        compound_pair_counts[sj] += 1

    df_compound_overrep = (
        pd.DataFrame(compound_pair_counts.items(), columns=["smiles", "n_pairs_involved"])
        .sort_values("n_pairs_involved", ascending=False)
        .reset_index(drop=True)
    )
    df_compound_overrep.to_csv(os.path.join(out_dir, f"{prefix}compound_pair_involvement.csv"), index=False)

    over_50 = df_compound_overrep[df_compound_overrep["n_pairs_involved"] >= overrep_cfg.threshold_50]
    over_100 = df_compound_overrep[df_compound_overrep["n_pairs_involved"] >= overrep_cfg.threshold_100]
    over_50.to_csv(os.path.join(out_dir, f"{prefix}compounds_over_{overrep_cfg.threshold_50}.csv"), index=False)
    over_100.to_csv(os.path.join(out_dir, f"{prefix}compounds_over_{overrep_cfg.threshold_100}.csv"), index=False)

    # 3) Distribution of scaffold frequencies (already have per-compound)
    # additionally: per-pair scaffold-pair frequency (core_key) distribution summary
    df_core_dist = df_core_freq["n_pairs"].value_counts().reset_index()
    df_core_dist.columns = ["n_pairs_per_core", "n_cores"]
    df_core_dist = df_core_dist.sort_values("n_pairs_per_core", ascending=False)
    df_core_dist.to_csv(os.path.join(out_dir, f"{prefix}pair_core_frequency_distribution.csv"), index=False)

    summary = {
        "n_pairs_total": int(len(df)),
        "n_pairs_with_scaffold_ok": int(n_pairs_ok),
        "n_unique_compounds": int(len(all_smiles)),
        "n_unique_scaffolds": int(len(scaffold_counts)),
        "n_unique_pair_cores": int(len(core_count)),
        "proportion_pairs_in_repeated_cores": float(proportion_pairs_repeated_core),
        "n_compounds_over_50_pairs": int(len(over_50)),
        "n_compounds_over_100_pairs": int(len(over_100)),
        "top10_scaffolds_by_compounds": df_scaffold_freq.head(10).to_dict(orient="records"),
        "top10_cores_by_pairs": df_core_freq.head(10).to_dict(orient="records"),
        "top10_compounds_by_pair_involvement": df_compound_overrep.head(10).to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, f"{prefix}summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def leakage_from_split_pairs(
    train_pairs_df: pd.DataFrame,
    val_pairs_df: pd.DataFrame,
    test_pairs_df: pd.DataFrame,
    out_dir: str,
    prefix: str = "",
) -> LeakageReport:
    """
    Checks compound overlap across splits and saves a report.
    """
    ensure_dir(out_dir)

    train_cmp = _extract_compounds_from_pairs_df(train_pairs_df)
    val_cmp = _extract_compounds_from_pairs_df(val_pairs_df)
    test_cmp = _extract_compounds_from_pairs_df(test_pairs_df)

    tv = sorted(train_cmp.intersection(val_cmp))
    tt = sorted(train_cmp.intersection(test_cmp))
    vt = sorted(val_cmp.intersection(test_cmp))

    rep = LeakageReport(
        train_val_overlap=len(tv),
        train_test_overlap=len(tt),
        val_test_overlap=len(vt),
        train_val_compounds=tv[:5000],
        train_test_compounds=tt[:5000],
        val_test_compounds=vt[:5000],
    )

    # Write a compact JSON summary + full lists as CSV (safe for big overlaps)
    with open(os.path.join(out_dir, f"{prefix}compound_overlap_summary.json"), "w") as f:
        json.dump(asdict(rep), f, indent=2)

    pd.DataFrame({"smiles": tv}).to_csv(os.path.join(out_dir, f"{prefix}overlap_train_val.csv"), index=False)
    pd.DataFrame({"smiles": tt}).to_csv(os.path.join(out_dir, f"{prefix}overlap_train_test.csv"), index=False)
    pd.DataFrame({"smiles": vt}).to_csv(os.path.join(out_dir, f"{prefix}overlap_val_test.csv"), index=False)

    return rep


def analyze_from_pairs_list(
    pairs_list,
    out_dir: str,
    overrep_cfg: OverrepConfig = OverrepConfig(),
    prefix: str = "",
) -> dict:
    df_pairs = _pairs_df_from_pairs_list(pairs_list)
    return analyze_pairs(df_pairs, out_dir, overrep_cfg=overrep_cfg, prefix=prefix)