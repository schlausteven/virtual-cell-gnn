import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from pathlib import Path

from tox21_dataset import get_dataloaders
from models.toxgnn import ToxGNN

from sklearn.metrics import roc_auc_score
import numpy as np

def eval_auc(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits = model(batch.x, batch.edge_index,
                           batch.edge_attr.float(), batch.batch)
            preds  = torch.sigmoid(logits).cpu()
            y_true.append(batch.y.cpu())
            y_pred.append(preds)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    aucs = []
    for t in range(12):
        m = (y_true[:, t] != -1) & (~torch.isnan(y_true[:, t]))
        if m.sum() > 0 and y_true[m, t].unique().numel() == 2:
            try:
                auc = roc_auc_score(y_true[m, t], y_pred[m, t])
                if not np.isnan(auc):
                    aucs.append(auc)
            except Exception as e:
                print(f"Error computing AUC for task {t}: {e}")
                continue
    if len(aucs) == 0:
        print("Warning: No valid AUCs computed, returning 0.5")
        return 0.5
    return float(torch.tensor(aucs).mean())


# ---------------------------------------------------------------------------
# BCE helper that ignores missing labels (targets == -1, nan...)
# ---------------------------------------------------------------------------

def masked_bce(logits: torch.Tensor,
               targets: torch.Tensor,
               pos_weight: torch.Tensor) -> torch.Tensor:
    mask = (targets != -1) & (~torch.isnan(targets))
    if mask.sum() == 0:
        return logits.sum() * 0.0           

    logits   = logits[mask].clamp(-20, 20)
    targets  = targets[mask].clamp(min=0).float()

    task_idx = mask.nonzero(as_tuple=False)[:, 1]      
    pw       = pos_weight[task_idx].to(logits.device)  

    prob  = torch.sigmoid(logits)
    eps   = logits.new_tensor(1e-7)

    element = -(pw * targets * torch.log(prob + eps) +
                (1 - targets) * torch.log(1 - prob + eps))
    return element.mean()


# ---------------------------------------------------------------------------
# Data & model
# ---------------------------------------------------------------------------
train_loader, val_loader, _, pos_w = get_dataloaders(batch_size=64, frac_train=0.7,    
    frac_val=0.2 )
DEVICE = torch.device("cpu")
pos_w  = pos_w.clamp_(min=1, max=15)            

model = ToxGNN(n_node_feats=9, n_edge_feats=3, n_tasks=12).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
MAX_GRAD_NORM = 1.0  

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
NUM_EPOCHS = 50
best_auc = 0.0
patience = 10
pat_ctr = 0

# Setup CSV logging
log_path = Path("log.csv")
if log_path.exists():
    log_path.unlink()  
with log_path.open("w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "val_auc", "lr"])

for epoch in range(NUM_EPOCHS):
    model.train()
    running = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        # Check for NaN in inputs before model forward pass
        if torch.isnan(batch.x).any():
            print(f"Epoch {epoch}, Batch {batch_idx}: NaN in node features!")
            continue
        if torch.isnan(batch.edge_attr).any():
            print(f"Epoch {epoch}, Batch {batch_idx}: NaN in edge features!")
            continue

        logits = model(batch.x, batch.edge_index, batch.edge_attr.float(), batch.batch)

        # Check for NaN in model output
        if torch.isnan(logits).any():
            print(f"Epoch {epoch}, Batch {batch_idx}: NaN in model output!")
            continue
        
        # Check for NaN in targets
        if torch.isnan(batch.y).all():
            print(f"Epoch {epoch}, Batch {batch_idx}: Batch has all NaN targets")
            continue
            
        loss_cpu = masked_bce(logits.cpu(), batch.y.cpu(), pos_w)
        
        if torch.isnan(loss_cpu):
            print(f"Epoch {epoch}, Batch {batch_idx}: NaN loss detected!")
            continue
            
        loss = loss_cpu.to(DEVICE)
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Epoch {epoch}, Batch {batch_idx}: NaN gradient in {name}")
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Skipping batch due to NaN gradients")
            optimizer.zero_grad()
            continue
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        running += loss.item()
        num_batches += 1

    # Check model parameters after training
    has_nan_params = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Epoch {epoch}: NaN in parameter after training: {name}")
            has_nan_params = True
    
    if has_nan_params:
        print("Model has NaN parameters, skipping evaluation")
        val_auc = 0.5
    else:
        val_auc = eval_auc(val_loader)

    avg_loss = running / num_batches if num_batches > 0 else float('inf')
    print(f"Epoch {epoch:02d} | mean_train_loss: {avg_loss:.4f} | val_auc: {val_auc:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Log to CSV
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow([epoch, f"{avg_loss:.4f}", f"{val_auc:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}"])
    
    # Update learning rate based on validation performance
    scheduler.step(val_auc)
    
    # Save best model
    if val_auc > best_auc + 1e-4:
        best_auc = val_auc
        pat_ctr = 0
        torch.save(model.state_dict(), "final_weights.pt")
        print(f"New best model saved with val_auc: {best_auc:.4f}")
    else:
        pat_ctr += 1
        if pat_ctr >= patience:
            print(f"Early stopping: val_AUC plateaued for {patience} epochs.")
            break

print(f"Training completed. Best validation AUC: {best_auc:.4f}")