import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_data(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Generate 2D coordinates for the molecule
    AllChem.Compute2DCoords(mol)
    
    # Get number of atoms
    num_atoms = mol.GetNumAtoms()
    
    # Create node features (atom features)
    x = []
    for atom in mol.GetAtoms():
        # Basic atom features
        atom_type = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        implicit_valence = atom.GetImplicitValence()
        aromatic = 1 if atom.GetIsAromatic() else 0
        hybridization = atom.GetHybridization()
        num_h = atom.GetTotalNumHs()
        radical_electrons = atom.GetNumRadicalElectrons()
        in_ring = 1 if atom.IsInRing() else 0
        
        # Normalize features to match Tox21 format
        features = [
            float(atom_type) / 100.0,  # Atomic number
            float(degree) / 10.0,      # Degree
            float(formal_charge) / 10.0,  # Formal charge
            float(implicit_valence) / 10.0,  # Implicit valence
            float(aromatic),           # Aromatic
            float(int(hybridization)) / 10.0,  # Hybridization
            float(num_h) / 10.0,       # Number of hydrogens
            float(radical_electrons) / 10.0,  # Radical electrons
            float(in_ring)             # In ring
        ]
        x.append(features)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Create edge features and connectivity
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        # Get atom indices
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Add both directions for undirected graph
        edge_index.append([start, end])
        edge_index.append([end, start])
        
        # Edge features (bond type, aromatic, conjugated, ring)
        bond_type = bond.GetBondType()
        is_aromatic = 1 if bond.GetIsAromatic() else 0
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        is_in_ring = 1 if bond.IsInRing() else 0
        
        # Normalize bond type
        bond_type_val = {
            Chem.BondType.SINGLE: 1.0,
            Chem.BondType.DOUBLE: 2.0,
            Chem.BondType.TRIPLE: 3.0,
            Chem.BondType.AROMATIC: 1.5
        }.get(bond_type, 1.0)
        
        edge_features = [
            bond_type_val / 3.0,  # Bond type
            float(is_aromatic),   # Aromatic
            float(is_in_ring)     # In ring
        ]
        
        # Add features for both directions
        edge_attr.append(edge_features)
        edge_attr.append(edge_features)
    
    if not edge_index:  # Handle single atoms
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create batch assignment (single graph)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )
    
    return data 