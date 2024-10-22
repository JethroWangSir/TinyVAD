import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, fbeta_score, roc_curve
import numpy as np
import json
from torchprofile import profile_macs
from torchinfo import summary
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.ndimage import median_filter

from data.dataset import AVA
from model.tinyvad import TinyVAD
from function.util import calculate_fpr_fnr

WINDOW_SIZE = 0.16
THRESHOLD = 0.5

# Set GPU and paths
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
font_prop = FontProperties(fname='/share/nas169/jethrowang/fonts/Times_New_Roman.ttf', size=13)
exp_dir = './exp/exp_0.16_tinyvad_old'
os.makedirs(exp_dir, exist_ok=True)

# Load test dataset and model
test_loader = DataLoader(AVA('/share/nas165/aaronelyu/Datasets/AVA-speech/', sample_duration=WINDOW_SIZE), batch_size=1)
if WINDOW_SIZE == 0.63:
    patch_size = 8
    frame_size = 64
elif WINDOW_SIZE == 0.32:
    patch_size = 4
    frame_size = 32
elif WINDOW_SIZE == 0.16:
    patch_size = 2
    frame_size = 16
elif WINDOW_SIZE == 0.025:
    patch_size = 1
    frame_size = 3
model = TinyVAD(1, 32, 64, patch_size, 2).to(device)
model.load_state_dict(torch.load(os.path.join(exp_dir, 'model_epoch_146_auroc=0.8272.ckpt')))
model.eval()

# Evaluation variables
val_labels_list, val_outputs_list, inference_times = [], [], []

# Test loop
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        val_inputs = [item[1].to(device) for item in batch]
        val_labels = [item[2].to(device).float().unsqueeze(1) for item in batch]

        val_inputs = torch.cat(val_inputs, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        # print(f'val_inputs: {val_inputs.shape}, val_labels: {val_labels.shape}')

        start_time = time.time()
        val_outputs = model(val_inputs)
        end_time = time.time()
        
        val_outputs_binary = (val_outputs > THRESHOLD).float().cpu().numpy()
        # print(f'val_outputs_binary: {val_outputs_binary}\nval_outputs_binary shape: {val_outputs_binary.shape}')
        
        if val_outputs_binary.any():
            val_scores = torch.tensor([[1.0]])
        else:
            val_scores = torch.tensor([[0.0]])

        val_outputs_list.append(val_scores)
        val_labels_list.append(val_labels[0].unsqueeze(0))

        # Record inference time
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
    
# Concatenate results
val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
val_outputs_cat = torch.cat(val_outputs_list, dim=0).cpu().numpy()

# Metrics calculation
auroc = roc_auc_score(val_labels_cat, val_outputs_cat)
binarized_preds = (val_outputs_cat >= THRESHOLD).astype(float)
accuracy = accuracy_score(val_labels_cat, binarized_preds)
f2_score = fbeta_score(val_labels_cat, binarized_preds, beta=2)
avg_inference_time = sum(inference_times) / len(inference_times)

# RTF calculation
rtf_list = []
for seconds in range(1, 11):
    random_input = torch.randn(1, 1, 64, 99 * seconds).to(device)
    start_time = time.time()
    model(random_input)
    end_time = time.time()
    inference_time = end_time - start_time
    rtf = inference_time / seconds
    rtf_list.append(rtf)

# Plot RTF
plt.plot(range(1, 11), rtf_list, marker='o', color='tab:red')
plt.xlabel('Duration (seconds)', fontproperties=font_prop)
plt.ylabel('Real-Time Factor (RTF)', fontproperties=font_prop)
plt.grid(True)
plt.savefig(os.path.join(exp_dir, 'rtf_plot_strict.png'), dpi=800)
plt.show()

# ROC curve calculation
fpr, tpr, thresholds = roc_curve(val_labels_cat, val_outputs_cat)

# Plot AUROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}', color='tab:blue')
plt.xlabel('False Positive Rate', fontproperties=font_prop)
plt.ylabel('True Positive Rate', fontproperties=font_prop)
plt.title('ROC Curve', fontproperties=font_prop)
plt.grid(True)

# Mark specific thresholds
thresholds_to_plot = np.arange(0.1, 1.1, 0.1)
for thr in thresholds_to_plot:
    # Find the closest threshold in ROC curve
    idx = np.argmin(np.abs(thresholds - thr))
    plt.plot(fpr[idx], tpr[idx], marker='o', label=f'Threshold={thr:.1f}')
plt.legend(prop=font_prop)
plt.savefig(os.path.join(exp_dir, 'auroc_plot_strict.png'), dpi=800)
plt.show()

# FPR/FNR calculation
fpr, fnr = calculate_fpr_fnr(val_labels_cat, val_outputs_cat)

# Save results and model information
results = {
    "AUROC": auroc, "Accuracy": accuracy, "F2-Score": f2_score, 
    "FPR": fpr, "FNR": fnr, "Avg Inference Time (ms)": avg_inference_time,
    "Real-Time Factor (RTF)": rtf_list
}
with open(os.path.join(exp_dir, 'test_strict.json'), 'w') as f:
    json.dump(results, f, indent=4)

params_count = summary(model, input_size=(1, 1, 64, frame_size), verbose=0).total_params / 1_000
macs = profile_macs(model, torch.randn(1, 1, 64, frame_size).to(device)) / 1_000_000
model_info = {"Param Count (k)": params_count, "MACs (M)": macs}
with open(os.path.join(exp_dir, 'model_info_strict.json'), 'w') as f:
    json.dump(model_info, f, indent=4)

# Print results
print(f"AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}, F2-Score: {f2_score:.4f}")
print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}, Avg Inference Time: {avg_inference_time:.4f} ms")
print(f"Param Count: {params_count:.2f}k, MACs: {macs:.2f}M")
