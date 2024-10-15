import os
import logging
import torch
from sklearn.metrics import confusion_matrix

# Learning rate scheduler with warmup, hold, and decay
def lr_lambda(current_step, total_steps, warmup_ratio, hold_ratio, learning_rate, min_lr):
    warmup_steps = total_steps * warmup_ratio
    hold_steps = total_steps * hold_ratio
    if current_step < warmup_steps:
        return float(current_step) / warmup_steps
    elif current_step < warmup_steps + hold_steps:
        return 1.0
    else:
        remaining_steps = total_steps - warmup_steps - hold_steps
        decay_step = current_step - warmup_steps - hold_steps
        return (1 - decay_step / remaining_steps) * (learning_rate - min_lr) / learning_rate + min_lr / learning_rate

# Function to save top-k models
def save_top_k_model_with_loss(exp_dir, model, epoch, val_loss, top_k_losses, k=3):
    if len(top_k_losses) < k:
        top_k_losses.append((val_loss, epoch))
        top_k_losses.sort(key=lambda x: x[0])
        model_path = os.path.join(exp_dir, f'model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.ckpt')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Model with val_loss {val_loss:.4f} saved to {model_path}')
    elif val_loss < top_k_losses[-1][0]:
        removed_loss, removed_epoch = top_k_losses.pop()
        old_model_path = os.path.join(exp_dir, f'model_epoch_{removed_epoch+1}_val_loss_{removed_loss:.4f}.ckpt')
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
            logging.info(f'Removed old model: {old_model_path}')
        top_k_losses.append((val_loss, epoch))
        top_k_losses.sort(key=lambda x: x[0])
        model_path = os.path.join(exp_dir, f'model_epoch_{epoch+1}_val_loss_{val_loss:.4f}.ckpt')
        torch.save(model.state_dict(), model_path)
        logging.info(f'Model with val_loss {val_loss:.4f} saved to {model_path}')

def save_top_k_model_with_auroc(exp_dir, model, epoch, auroc, top_k_auc_scores, k=3):
    checkpoint_path = os.path.join(exp_dir, f'model_epoch_{epoch + 1}_auroc={auroc:.4f}.ckpt')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Model saved to {checkpoint_path}')

    top_k_auc_scores.append((auroc, checkpoint_path))
    top_k_auc_scores.sort(key=lambda x: x[0], reverse=True)

    if len(top_k_auc_scores) > k:
        _, worst_model_path = top_k_auc_scores.pop()
        if os.path.exists(worst_model_path):
            os.remove(worst_model_path)
            logging.info(f'Removed model: {worst_model_path}')

# Function to calculate FPR and FNR
def calculate_fpr_fnr(y_true, y_pred, threshold=0.5):
    # Apply threshold to predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Calculate FPR and FNR
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    return fpr, fnr