"""tox21_gui.py
Run this inside a Jupyter notebook cell (`%run tox21_gui.py`) or just copy‑paste
cells into a fresh notebook.  It creates a simple SMILES‑to‑toxicity GUI using
ipywidgets and RDKit’s SVG renderer.

REQUIREMENTS
------------
conda install -c conda-forge ipywidgets rdkit-pypi matplotlib

USAGE
-----
1. Start JupyterLab:  `jupyter lab`
2. In a new Python 3 notebook run:  `%run tox21_gui.py`
3. A text box + “Predict” button appear; paste a SMILES, click Predict.
"""
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Draw
import ipywidgets as widgets
from IPython.display import display, clear_output
from rdkit import RDLogger

from models.toxgnn import ToxGNN
from utils import smiles_to_data  # helper we wrote earlier

RDLogger.DisableLog("rdApp.*")

TASK_NAMES = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Arom","NR-ER","NR-ER-LBD",
    "NR-PPAR-γ","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]

# ---------------- load the trained model --------------------------
model = ToxGNN(n_node_feats=9, n_edge_feats=3, n_tasks=12)
model.load_state_dict(torch.load("final_weights.pt", map_location="cpu"))
model.eval()

# ---------------- widgets -----------------------------------------
smiles_box = widgets.Text(
    description="SMILES:",
    value="CC1=CC=C(C=C1)O",  # phenol as default
    placeholder="Paste a SMILES string"
)
button = widgets.Button(description="Predict", button_style="success")
output = widgets.Output()

# ---------------- callback ----------------------------------------

def on_click(_):
    with output:
        clear_output()
        smiles = smiles_box.value.strip()
        try:
            data = smiles_to_data(smiles)
        except Exception as e:
            print("⚠️  RDKit error:", e)
            return

        # inference
        loader = DataLoader([data], batch_size=1)
        with torch.no_grad():
            batch = next(iter(loader))
            logits = model(batch.x, batch.edge_index,
                            batch.edge_attr.float(), batch.batch)
            probs = torch.sigmoid(logits).squeeze(0).tolist()

        # show molecule SVG
        mol = Chem.MolFromSmiles(smiles)
        display(Draw.MolToImage(mol, size=(200,200)))

        # print table
        for name, p in zip(TASK_NAMES, probs):
            print(f"{name:10s}: {p:.3f}")

button.on_click(on_click)

display(widgets.VBox([smiles_box, button, output]))
