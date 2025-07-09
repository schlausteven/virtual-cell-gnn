# Tox21 GNN Model

A Graph Neural Network (GNN) model for predicting molecular toxicity using the Tox21 dataset. This model can predict 12 different toxicity endpoints for chemical compounds.

## Overview

The Tox21 dataset contains ~8,000 compounds tested for 12 different toxicity endpoints including nuclear receptor and stress response pathways. This project implements a GNN-based approach using PyTorch Geometric to predict these toxicity properties from molecular SMILES strings.

## Model Architecture

The model uses a Graph Isomorphism Network (GIN) with edge features:
- **Node features**: 9 atom-level features (atomic number, degree, formal charge, etc.)
- **Edge features**: 3 bond-level features (bond type, aromaticity, ring membership)
- **Architecture**: 3 GIN layers (64 → 128 → 256) + global pooling + MLP head
- **Output**: 12 binary classification tasks (toxicity endpoints)

## Training Process

The model was trained with the following approach:

### Data Preparation
- **Dataset**: Tox21 from MoleculeNet
- **Split**: Bemis-Murcko scaffold split (70% train, 20% val, 10% test)
- **Features**: RDKit-generated molecular graph features
- **Labels**: Binary toxicity endpoints with missing value handling

### Training Configuration
- **Optimizer**: AdamW with learning rate 1e-4
- **Loss**: Binary cross-entropy with positive class weighting
- **Regularization**: Dropout (0.1-0.3), weight decay 1e-4, gradient clipping
- **Batch size**: 64
- **Epochs**: 50 with early stopping (patience=10)

### Performance
- **Validation AUC**: 0.714
- **Training time**: ~30 minutes on CPU
- **Model size**: ~530KB

## Installation

```bash
# Install PyTorch and PyTorch Geometric
pip install torch torch-geometric

# Install RDKit and other dependencies
conda install -c conda-forge rdkit-pypi
pip install scikit-learn numpy

# For GUI (optional)
conda install -c conda-forge ipywidgets matplotlib
```

## Usage

### Training

To train the model from scratch:

```bash
python train.py
```

This will:
- Download the Tox21 dataset automatically
- Train the model with the optimized configuration
- Save the best model as `final_weights.pt`
- Log training progress to `log.csv`

### Testing

To test the model with various molecules:

```bash
python test_model.py
```

This tests the model on 8 different molecules and provides analysis of:
- Prediction variance between molecules
- Model collapse detection
- Extreme prediction analysis

### GUI Interface

The easiest way to use the model is through the Jupyter GUI:

1. Start Jupyter:
```bash
jupyter lab
```

2. Create a new notebook and run:
```python
%run tox21_gui.py
```

3. A GUI will appear with:
   - Text box for SMILES input
   - "Predict" button
   - Molecule visualization
   - Toxicity predictions for all 12 endpoints

### Example Usage

```python
# Load model
from models.toxgnn import ToxGNN
from utils import smiles_to_data
from torch_geometric.loader import DataLoader

model = ToxGNN(n_node_feats=9, n_edge_feats=3, n_tasks=12)
model.load_state_dict(torch.load('final_weights.pt', map_location='cpu'))
model.eval()

# Predict toxicity for a molecule
smiles = "CC1=CC=C(C=C1)O"  # phenol
data = smiles_to_data(smiles)
loader = DataLoader([data], batch_size=1)
batch = next(iter(loader))

with torch.no_grad():
    logits = model(batch.x, batch.edge_index, batch.edge_attr.float(), batch.batch)
    probs = torch.sigmoid(logits).squeeze(0).tolist()

# Print predictions
task_names = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Arom", "NR-ER", "NR-ER-LBD",
              "NR-PPAR-γ", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

for name, prob in zip(task_names, probs):
    print(f"{name}: {prob:.3f}")
```

## Toxicity Endpoints

The model predicts 12 different toxicity endpoints:

**Nuclear Receptor (NR) Pathways:**
- NR-AR: Androgen receptor
- NR-AR-LBD: Androgen receptor ligand binding domain
- NR-AhR: Aryl hydrocarbon receptor
- NR-Aromatase: Aromatase
- NR-ER: Estrogen receptor
- NR-ER-LBD: Estrogen receptor ligand binding domain
- NR-PPAR-γ: Peroxisome proliferator-activated receptor gamma

**Stress Response (SR) Pathways:**
- SR-ARE: Antioxidant response element
- SR-ATAD5: ATAD5
- SR-HSE: Heat shock response element
- SR-MMP: Matrix metalloproteinase
- SR-p53: p53

## File Structure

```
├── train.py              # Training script
├── eval.py               # Model testing script
├── models/
│   ├── toxgnn.py         # GNN model architecture
│   └── __init__.py
├── tox21_dataset.py      # Dataset loading and splitting
├── utils.py              # SMILES to graph conversion
├── tox21_gui.py          # Interactive GUI
├── final_weights.pt      # Trained model weights
├── log.csv               # Training logs
└── README.md
```

## Model Validation

The model has been validated to:
- Give different predictions for different molecules
- Avoid model collapse (no identical predictions)
- Provide reasonable confidence levels
- Handle various molecular structures correctly

## Limitations

- Trained on Tox21 dataset only (~8K compounds)
- Binary classification (active/inactive)
- May not generalize to all chemical space
- Requires valid SMILES strings as input

## Citation

If you use this code, please cite:
- Tox21 dataset: [Tox21 Challenge](https://tripod.nih.gov/tox21/challenge/)
- PyTorch Geometric: [Fey & Lenssen, 2019](https://arxiv.org/abs/1903.02428) 