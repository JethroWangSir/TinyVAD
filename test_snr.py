import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import json
from data.dataset import AVA_ESC
from tqdm import tqdm
from model.tinyvad import TinyVAD
from scipy.ndimage import median_filter

WINDOW_SIZE = 0.63
MEDIAN_KERNEL_SIZE = 63
THRESHOLD = 0.5

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name = 'exp1_tinyvad'
exp_dir = f'./exp/{name}/'
os.makedirs(exp_dir, exist_ok=True)

# SNR values to test
snr_list = [10, 5, 0]

# Initialize model and load the best checkpoint
if WINDOW_SIZE == 0.63:
    patch_size = 8
elif WINDOW_SIZE == 0.32:
    patch_size = 4
elif WINDOW_SIZE == 0.16:
    patch_size = 2
elif WINDOW_SIZE == 0.025:
    patch_size = 1
model = TinyVAD(1, 32, 64, patch_size, 2).to(device)
checkpoint_path = os.path.join(exp_dir, 'model.ckpt')
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# Variables for storing evaluation metrics
results = {}

for snr in snr_list:
    # Load test dataset with the current SNR
    test_dataset = AVA_ESC(
        ava_dir='/share/nas165/aaronelyu/Datasets/AVA-speech/',
        esc_csv='./data/ESC-50/esc50.csv',
        esc_audio_dir='./data/ESC-50/audio/',
        snr=snr,
    )
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Variables for storing predictions and true labels
    val_labels_list = []
    val_outputs_list = []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Testing SNR={snr}'):
            val_inputs = [item[0].to(device) for item in batch]
            val_labels = [item[1].to(device).float().unsqueeze(1) for item in batch]

            val_inputs = torch.cat(val_inputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            val_outputs = model(val_inputs)

            # Apply median filter to each batch's predictions
            val_outputs_np = val_outputs.cpu().numpy()
            val_outputs_smoothed = torch.tensor(median_filter(val_outputs_np, size=(MEDIAN_KERNEL_SIZE, 1))).to(device)

            val_labels_list.append(val_labels)
            val_outputs_list.append(val_outputs_smoothed)

    # Concatenate results
    val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
    val_outputs_cat = torch.cat(val_outputs_list, dim=0).cpu().numpy()

    # Compute AUROC
    auroc = roc_auc_score(val_labels_cat, val_outputs_cat)

    # Compute FPR and FNR
    binarized_preds = torch.tensor(val_outputs_cat >= THRESHOLD).float().cpu().numpy()
    tn, fp, fn, tp = confusion_matrix(val_labels_cat, binarized_preds).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    results[snr] = {
        'AUROC': auroc,
        'FPR': fpr,
        'FNR': fnr
    }

    print(f"SNR={snr}, AUROC: {auroc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")

# Save results to JSON
results_file = os.path.join(exp_dir, 'test_snr.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

# Print overall results
print("All evaluation results:", results)
