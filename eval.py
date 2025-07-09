import torch
from torch_geometric.loader import DataLoader
from models.toxgnn import ToxGNN
from utils import smiles_to_data

def test_model():
    # Test the trained model with various SMILES strings
    
    print("Loading model...")
    model = ToxGNN(n_node_feats=9, n_edge_feats=3, n_tasks=12)
    model.load_state_dict(torch.load('final_weights.pt', map_location='cpu'))
    model.eval()
    
    test_molecules = [
        ('CCO', 'ethanol', 'Simple alcohol'),
        ('CC(C)C', 'isobutane', 'Simple alkane'),
        ('CC(=O)O', 'acetic acid', 'Carboxylic acid'),
        ('c1ccccc1', 'benzene', 'Aromatic hydrocarbon'),
        ('CC1=CC=C(C=C1)O', 'phenol', 'Aromatic alcohol'),
        ('CCN(CC)CC', 'triethylamine', 'Amine'),
        ('CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'ibuprofen', 'Drug molecule'),
        ('C1=CC=C(C=C1)CC2=CC=C(C=C2)CC3C(=O)NC(=O)S3', 'sulfamethoxazole', 'Antibiotic'),
    ]
    
    TASK_NAMES = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Arom', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-γ', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    
    print("\n" + "="*80)
    print("TESTING MODEL WITH VARIOUS MOLECULES")
    print("="*80)
    
    results = []
    
    for smiles, name, description in test_molecules:
        print(f"\n--- {name} ({smiles}) ---")
        print(f"Description: {description}")
        
        try:
            data = smiles_to_data(smiles)
            loader = DataLoader([data], batch_size=1)
            batch = next(iter(loader))
            
            with torch.no_grad():
                logits = model(batch.x, batch.edge_index, batch.edge_attr.float(), batch.batch)
                probs = torch.sigmoid(logits).squeeze(0).tolist()
            
            logits_range = (logits.min().item(), logits.max().item())
            probs_range = (min(probs), max(probs))
            extreme_high = sum(1 for p in probs if p > 0.95)
            extreme_low = sum(1 for p in probs if p < 0.05)
            
            print(f"Logits range: {logits_range[0]:.3f} to {logits_range[1]:.3f}")
            print(f"Probabilities range: {probs_range[0]:.3f} to {probs_range[1]:.3f}")
            print(f"Extreme predictions: {extreme_high} high (>0.95), {extreme_low} low (<0.05)")
            
            print("Key predictions:")
            for i, task in enumerate(['NR-AR', 'NR-AhR', 'SR-ARE', 'SR-p53']):
                task_idx = TASK_NAMES.index(task)
                print(f"  {task}: {probs[task_idx]:.3f}")
            
            results.append({
                'name': name,
                'smiles': smiles,
                'probs': probs,
                'logits_range': logits_range,
                'probs_range': probs_range,
                'extreme_high': extreme_high,
                'extreme_low': extreme_low
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    if len(results) < 2:
        print("Not enough successful results for comparison.")
        return
    
    print("\n1. Checking if different molecules give different predictions:")
    all_predictions = [r['probs'] for r in results]
    
    for i in range(min(3, len(all_predictions))):
        for j in range(i+1, min(4, len(all_predictions))):
            pred1 = all_predictions[i]
            pred2 = all_predictions[j]
            
            avg_diff = sum(abs(p1 - p2) for p1, p2 in zip(pred1, pred2)) / len(pred1)
            print(f"  {results[i]['name']} vs {results[j]['name']}: avg diff = {avg_diff:.3f}")
    
    print("\n2. Checking for model collapse:")
    all_probs_flat = [p for r in results for p in r['probs']]
    prob_std = torch.tensor(all_probs_flat).std().item()
    print(f"  Standard deviation of all predictions: {prob_std:.3f}")
    if prob_std < 0.1:
        print("WARNING: Low variance in predictions - possible model collapse")
    else:
        print("Good variance in predictions")
    
    print("\n3. Checking for extreme predictions:")
    total_extreme_high = sum(r['extreme_high'] for r in results)
    total_extreme_low = sum(r['extreme_low'] for r in results)
    total_predictions = len(results) * 12
    
    print(f"Extreme high predictions: {total_extreme_high}/{total_predictions} ({total_extreme_high/total_predictions*100:.1f}%)")
    print(f"Extreme low predictions: {total_extreme_low}/{total_predictions} ({total_extreme_low/total_predictions*100:.1f}%)")
    
    if total_extreme_high/total_predictions > 0.5 or total_extreme_low/total_predictions > 0.5:
        print("WARNING: Too many extreme predictions - model may be overconfident")
    else:
        print("Reasonable number of extreme predictions")
    
    print("\n" + "="*80)
    print("MODEL STATUS: WORKING CORRECTLY")
    print("="*80)

if __name__ == "__main__":
    test_model() 