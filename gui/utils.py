"""Utility functions for the Tox21 GNN project."""
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_data(smiles: str) -> Data:
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    AllChem.Compute2DCoords(mol)
    
    num_atoms = mol.GetNumAtoms()
    
    x = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        implicit_valence = atom.GetImplicitValence()
        aromatic = 1 if atom.GetIsAromatic() else 0
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()
        radical_electrons = atom.GetNumRadicalElectrons()
        in_ring = 1 if atom.IsInRing() else 0
        
        features = [
            float(atom_type) / 100.0,
            float(degree) / 10.0,
            float(formal_charge) / 10.0,
            float(implicit_valence) / 10.0,
            float(aromatic),
            float(int(hybridization)) / 10.0,
            float(num_h) / 10.0,
            float(radical_electrons) / 10.0,
            float(in_ring)
        ]
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        edge_index.append([start, end])
        edge_index.append([end, start])
        
        bond_type = bond.GetBondType()
        is_aromatic = 1 if bond.GetIsAromatic() else 0
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        is_in_ring = 1 if bond.IsInRing() else 0
        
        bond_type_val = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5
        }.get(bond_type, 1.0)
        
        edge_features = [
            bond_type_val / 3.0,
            float(is_aromatic),
            float(is_in_ring)
        ]
        
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)
    
    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    batch = torch.zeros(num_atoms, dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )
    
    return data 