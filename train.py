import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from data.dataset import SCF, SCF_ESC, AVA
from model.tinyvad import TinyVAD
from function.roc_star import epoch_update_gamma, roc_star_loss
from function.util import lr_lambda, save_top_k_model_with_auroc, calculate_fpr_fnr

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name = f'exp_0.16_esc_tinyvad'
exp_dir = f'./exp/{name}/'
os.makedirs(exp_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="TinyVAD", name=name, config={
    "seed": 42,
    "epochs": 150,
    "batch_size": 256,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.001,
    "warmup_ratio": 0.05,
    "hold_ratio": 0.45,
    "min_lr": 0.001,
    "augment": True,
    "roc_star": False,
    "roc_star_weight": 0.0,
    "train_with_esc": True,
    "window_size": 0.16,
})
config = wandb.config

torch.manual_seed(config.seed)

# Setup logging
log_file = os.path.join(exp_dir, 'train.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# Load datasets
train_manifests = ['./data/manifest/balanced_background_training_manifest.json', './data/manifest/balanced_speech_training_manifest.json']
val_dir = '/share/nas165/aaronelyu/Datasets/AVA-speech/'
logging.info('Loading training set...')
if config.train_with_esc:
    train_dataset = SCF_ESC(
        manifest_files=train_manifests,
        noise_csv='./data/ESC-50/esc50.csv',
        noise_audio_dir='./data/ESC-50/audio/',
        sample_duration=config.window_size,
        augment=config.augment,
    )
else:
    train_dataset = SCF(train_manifests, sample_duration=config.window_size, augment=config.augment)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
logging.info(f'Training set size: {len(train_loader)}')
logging.info('Loading validation set...')
val_dataset = AVA(root_dir=val_dir, sample_duration=config.window_size)
val_loader = DataLoader(val_dataset, batch_size=1)
logging.info(f'Validation set size: {len(val_loader)}')
logging.info('Finish loading dataset!')
print('------------------------------')

# Initialize model, loss function, and optimizer
if config.window_size == 0.63:
    patch_size = 8
elif config.window_size == 0.32:
    patch_size = 4
elif config.window_size == 0.16:
    patch_size = 2
elif config.window_size == 0.025:
    patch_size = 1
model = TinyVAD(1, 32, 64, patch_size, 2).to(device)
bce_criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, len(train_loader) * config.epochs, config.warmup_ratio, config.hold_ratio, config.learning_rate, config.min_lr))

epoch_gamma = 0.20
top_3_auc_scores = []

# Training loop
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0

    all_labels = []
    all_preds = []

    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch [{epoch + 1}/{config.epochs}] Training')

    for batch_idx, batch in train_progress_bar:
        if config.window_size == 0.63:
            inputs, labels = batch[0].to(device), batch[1].to(device).float().unsqueeze(1)
        else:
            inputs = [item[0].to(device) for item in batch]
            labels = [item[1].to(device).float().unsqueeze(1) for item in batch]
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)

        optimizer.zero_grad()
        outputs = model(inputs)

        if epoch > 4 and config.roc_star:
            if batch_idx == 0: print('*Using Loss Roc-star and BCE')
            loss1 = config.roc_star_weight * roc_star_loss(labels, outputs, epoch_gamma, last_all_labels, last_all_preds)
            loss2 = (1 - config.roc_star_weight) * bce_criterion(outputs, labels)
            wandb.log({"roc_star_loss": loss1.item(), "bce_loss": loss2.item()})
            loss = loss1 + loss2

        else:
            if batch_idx == 0: print('*Using Loss BCE')
            loss = bce_criterion(outputs, labels)
            wandb.log({"bce_loss": loss.item()})

        loss.backward()

        if epoch > 4 and config.roc_star:
            # To prevent gradient explosions resulting in NaNs
            # https://discuss.pytorch.org/t/nan-loss-in-rnn-model/655/8
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            wandb.log({"train_loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]})

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(outputs.detach().cpu().numpy())

    avg_train_loss = running_loss / len(train_loader)
    logging.info(f'Average Train Loss: {avg_train_loss:.4f}')
    wandb.log({"avg_train_loss": avg_train_loss})

    last_all_labels = torch.tensor(all_labels).to(device)
    last_all_preds = torch.tensor(all_preds).to(device)

    # Validation step
    model.eval()

    val_labels_list, val_outputs_list = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            val_inputs = [item[1].to(device) for item in batch]
            val_labels = [item[2].to(device).float().unsqueeze(1) for item in batch]

            val_inputs = torch.cat(val_inputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            val_outputs = model(val_inputs)
            
            val_outputs_avg = val_outputs.mean().unsqueeze(0)
            val_outputs_list.append(val_outputs_avg)

            val_labels_list.append(val_labels[0].unsqueeze(0))

    # Concatenate results
    val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
    val_outputs_cat = torch.cat(val_outputs_list, dim=0).cpu().numpy()

    # Metrics calculation
    auroc = roc_auc_score(val_labels_cat, val_outputs_cat)
    wandb.log({"auroc": auroc, "epoch": epoch + 1})

    # Calculate FPR and FNR
    fpr, fnr = calculate_fpr_fnr(val_labels_cat, val_outputs_cat)
    wandb.log({"fpr": fpr, "fnr": fnr})

    logging.info(f'Epoch [{epoch + 1}/{config.epochs}]: AUROC = {auroc:.4f}, FPR = {fpr:.4f}, FNR = {fnr:.4f}')

    # Update gamma for the next epoch
    if config.roc_star_loss:
        epoch_gamma = epoch_update_gamma(last_all_labels, last_all_preds, epoch)

    # Save the top 3 AUC models
    save_top_k_model_with_auroc(exp_dir, model, epoch, auroc, top_3_auc_scores, k=3)

# After last epoch, save final model
final_checkpoint = os.path.join(exp_dir, f'model_last_epoch.ckpt')
torch.save(model.state_dict(), final_checkpoint)
logging.info(f'Final model saved to {final_checkpoint}')

logging.info('Training complete!')
wandb.finish()
