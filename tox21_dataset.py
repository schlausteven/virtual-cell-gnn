from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

SEED: int = 42
random.seed(SEED)
torch.manual_seed(SEED)

RAW_DIR = Path.home() / ".cache" / "tox21"
TASK_NAMES: List[str] = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

def _generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def _scaffold_split(dataset, frac_train: float = 0.8, frac_val: float = 0.1
                    ) -> Tuple[List[int], List[int], List[int]]:
    scaff2indices: dict[str, List[int]] = {}
    for idx, data in enumerate(dataset):
        scaffold = _generate_scaffold(data.smiles)
        scaff2indices.setdefault(scaffold, []).append(idx)

    buckets = sorted(scaff2indices.values(), key=len, reverse=True)

    n_total = len(dataset)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)

    train_idx, val_idx, test_idx = [], [], []
    for bucket in buckets:
        if len(train_idx) + len(bucket) <= n_train:
            train_idx += bucket
        elif len(val_idx) + len(bucket) <= n_val:
            val_idx += bucket
        else:
            test_idx += bucket
    return train_idx, val_idx, test_idx

def compute_pos_weights(dataset, indices: List[int]) -> torch.Tensor:
    y = dataset.data.y[indices]
    pos_counts = (y == 1).sum(dim=0).float()
    neg_counts = (y == 0).sum(dim=0).float()
    return neg_counts / (pos_counts + 1e-6)

def get_dataloaders(batch_size: int = 128, frac_train: float = 0.8,
                    frac_val: float = 0.1, seed: int = SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = MoleculeNet(root=str(RAW_DIR), name="Tox21")

    if not hasattr(dataset[0], "smiles"):
        for data, smi in zip(dataset, dataset.smiles):
            data.smiles = smi

    train_idx, val_idx, test_idx = _scaffold_split(dataset, frac_train, frac_val)

    def _non_empty(idxs):
        return [i for i in idxs if dataset[i].x.numel() > 0]

    train_idx, val_idx, test_idx = map(_non_empty, (train_idx, val_idx, test_idx))

    loaders = [
        DataLoader(dataset[idxs], batch_size=batch_size, shuffle=shuf)
        for idxs, shuf in zip((train_idx, val_idx, test_idx), (True, False, False))
    ]

    pos_weights = compute_pos_weights(dataset, train_idx)
    pos_weights = pos_weights.clamp(max=100)
    return *loaders, pos_weights

if __name__ == "__main__":
    train_loader, val_loader, test_loader, pos_w = get_dataloaders()
    print("Train batches:", len(train_loader))
    print("Val batches  :", len(val_loader))
    print("Test batches :", len(test_loader))
    print("pos_weights  :", pos_w)
