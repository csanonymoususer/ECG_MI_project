from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score
)
import matplotlib.pyplot as plt
import torch
import numpy as np
import wfdb

def plot_roc_curves(trainer, loaders_dict):

    colors = {'All': 'blue', 'Male': 'green', 'Female': 'orange'}

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=0.8)

    for name, loader in loaders_dict.items():
        trainer.model.eval()
        all_probs, all_targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = trainer._move_to_device(batch)
                x, targets = batch
                logits = trainer.model(x)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc = roc_auc_score(all_targets, all_probs)

        plt.plot(fpr, tpr, color=colors[name], linewidth=2,
                 label=f'{name}  (AUC = {auc:.3f})')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('ROC curves by sex', fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()


def specificity(loader, model, threshold=0.5):
    all_targets = []
    all_predicted = []
    model.eval()
    for inputs, targets in loader:
        with torch.no_grad():
            outputs = model(inputs.to(next(model.parameters()).device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.numpy()
            predicted = (probs>threshold)
            all_targets.extend(targets)
            all_predicted.extend(predicted)
            
    score = recall_score(targets, predicted, pos_label=0)
    return score


def sensitivity(loader, model, threshold=0.5):
    all_targets = []
    all_predicted = []
    model.eval()
    for inputs, targets in loader:
        with torch.no_grad():
            outputs = model(inputs.to(next(model.parameters()).device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.numpy()
            predicted = (probs>threshold)
            all_targets.extend(targets)
            all_predicted.extend(predicted)
            
    score = recall_score(targets, predicted)
    return score


def accuracy(loader, model, threshold=0.5):
    correct = 0
    total = 0
    model.eval()
    for inputs, targets in loader:
        with torch.no_grad():
            outputs = model(inputs.to(next(model.parameters()).device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.numpy()
            predicted = (probs>threshold)
            total += targets.size
            correct += (predicted == targets).sum().item()
    return correct / total


def roc_auc(loader, model):
    all_targets = []
    all_predicted = []
    model.eval()
    for inputs, targets in loader:
        with torch.no_grad():
            outputs = model(inputs.to(next(model.parameters()).device))
            probs = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.numpy()
            all_targets.extend(targets)
            all_predicted.extend(probs)
    score = roc_auc_score(targets, probs)
    return score



def plot_ecg_comparison(dataset, lead=1):
    df = dataset.train_ds.frame
    
    idx_norm_m = df[(df['has_mi'] == False) & (df['sex'] == 0)].index[0]
    idx_norm_f = df[(df['has_mi'] == False) & (df['sex'] == 1)].index[0]
    idx_mi_m = df[(df['has_mi'] == True)  & (df['sex'] == 0)].index[0]
    idx_mi_f = df[(df['has_mi'] == True)  & (df['sex'] == 1)].index[0]

    groups = [
        (idx_norm_m, 'Male — NORM'),
        (idx_norm_f, 'Female — NORM'),
        (idx_mi_m, 'Male — MI'),
        (idx_mi_f, 'Female — MI'),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for ax, (idx, title) in zip(axes, groups):
        row = df.loc[idx]
        signal, _ = wfdb.rdsamp(dataset.data_path + row['filename_hr'])
        sig = signal[:, lead]
        
        
        ax.plot(sig, linewidth=0.8)
        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(-4, 4)
        ax.grid(True, alpha=0.3)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Sample (100 Hz)', fontsize=10)
    fig.suptitle(f'ECG comparison — Lead {lead}', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
