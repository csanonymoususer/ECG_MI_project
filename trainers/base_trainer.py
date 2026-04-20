import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score,
    precision_score, f1_score, confusion_matrix,
    classification_report, roc_curve
)
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = config["experiment"]["device"]
        self.threshold = float(config["metrics"]["threshold"])
        self.es_enabled = config["early_stopping"]["enabled"]
        self.es_monitor = config["early_stopping"]["monitor"]
        self.es_mode = config["early_stopping"]["mode"]
        self.es_patience = int(config["early_stopping"]["patience"])
        self.epochs = int(config["training"]["epochs"])
        self.save_interval = int(config["checkpoints"]["save_every"])
        self.best_metric = None
        self.es_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_accuracy": [],
            "val_sensitivity": [],
            "val_specificity": []
        }

    def train(self, train_loader, val_loader):
        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch

            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate(val_loader)

            print(f"Epoch {epoch}")
            print(f"Train loss: {train_loss:.4f}", f"Val loss: {val_loss:.4f}")
            print(f"Val AUC: {metrics['auc']:.4f}", f"Val accuracy: {metrics['accuracy']:.4f}", f"Val sensitivity: {metrics['sensitivity']:.4f}", f"Val specificity: {metrics['specificity']:.4f}")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(metrics["auc"])
            self.history["val_accuracy"].append(metrics["accuracy"])
            self.history["val_sensitivity"].append(metrics["sensitivity"])
            self.history["val_specificity"].append(metrics["specificity"])

            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint()

            if self.scheduler is not None:
                self.scheduler.step()
            

            if self.es_enabled:
                if self._check_early_stopping():
                    break
            
 
    def _move_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) for b in batch]
        return batch.to(self.device)
    
    def _check_early_stopping(self):
        current = self.history[self.es_monitor][-1]

        if self.best_metric is None:
            self.best_metric = current
            return False 

        if self.es_mode == "min":
            improved = current < self.best_metric
        else:
            improved = current > self.best_metric

        if improved:
            self.best_metric = current
            self.es_counter = 0
        else:
            self.es_counter += 1

        if self.es_counter >= self.es_patience:
            print("Early stopping triggered")
            return True
        return False

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(loader):
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad()

            loss, logits, targets = self.compute_loss(batch)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()

        all_logits = []
        all_targets = []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = self._move_to_device(batch)

                loss, logits, targets = self.compute_loss(batch)

                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        metrics = self.compute_metrics(all_logits, all_targets)

        return total_loss / len(loader), metrics

    def compute_loss(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y.float())
        return loss, logits, y
        

    def compute_metrics(self, outputs, targets):
        probs = torch.sigmoid(outputs).numpy()
        targets = targets.numpy()

        preds = (probs > self.threshold).astype(int)

        return {
            "auc": roc_auc_score(targets, probs),
            "accuracy": accuracy_score(targets, preds),
            "sensitivity": recall_score(targets, preds),           # recall на MI
            "specificity": recall_score(targets, preds, pos_label=0)  # recall на NORM
        }
    
    def choose_threshold(self, loader):
        self.model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = self._move_to_device(batch)
                _, logits, targets = self.compute_loss(batch)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                targets = targets.cpu().numpy()
                
                all_probs.extend(probs)
                all_targets.extend(targets)
        
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
        
        idx = np.where(tpr >= 0.90)[0]
        best_thresh = thresholds[idx[np.argmax(tpr[idx] - fpr[idx])]]
        
        print(f"Best threshold: {best_thresh:.3f}")
        print(f"Sensitivity: {tpr[idx[np.argmax(tpr[idx] - fpr[idx])]]:.3f}")
        print(f"Specificity: {1 - fpr[idx[np.argmax(tpr[idx] - fpr[idx])]]:.3f}")
        print(f"AUC: {roc_auc_score(all_targets, all_probs):.3f}")
        print(f"Accuracy: {accuracy_score(all_targets, (all_probs > best_thresh).astype(int)):.3f}")
        
        self.threshold = best_thresh
        return best_thresh

        

    def save_checkpoint(self):
        name = self.config["experiment"]["name"]
        dir = self.config["checkpoints"]["checkpoint_dir"]+f"/{name}"
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = dir+ f"/model_epoch{self.epoch}.pt"
        torch.save(self.model.state_dict(), path)
        

    def load_checkpoint(self):
        path = self.config["checkpoints"]["checkpoint_path_for_load"]
        if not path:
            raise ValueError("No checkpoint path provided")
        self.model.load_state_dict(torch.load(path))

    def plot_history(self, save=True):
        epochs = range(1, len(self.history["train_loss"]) + 1)


        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves")
        plt.legend()


        plt.subplot(1,2,2)
        plt.plot(epochs, self.history["val_auc"], label="Val AUC")
        plt.plot(epochs, self.history["val_accuracy"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation Metrics")
        plt.legend()

        plt.tight_layout()
        if save:
            name = self.config["experiment"]["name"]
            dir = self.config["logging"]["log_dir"]+f"/{name}"
            if not os.path.exists(dir):
                os.makedirs(dir)
            plt.savefig(dir+f"/training_history.png")

        plt.show()

    def test(self, loader, threshold=None):
        if threshold is None:
            threshold = self.threshold
        
        self.model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                batch = self._move_to_device(batch)
                _, logits, targets = self.compute_loss(batch)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > threshold).astype(int)

        auc = roc_auc_score(all_targets, all_probs)
        acc = accuracy_score(all_targets, preds)
        sens = recall_score(all_targets, preds)
        spec = recall_score(all_targets, preds, pos_label=0)
        prec = precision_score(all_targets, preds)
        f1 = f1_score(all_targets, preds)
        cm = confusion_matrix(all_targets, preds)

        tn, fp, fn, tp = cm.ravel()

        print("=" * 50)
        print("           TEST RESULTS")
        print("=" * 50)
        print(f"  Threshold         : {threshold:.3f}")
        print(f"  AUC               : {auc:.4f}")
        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Sensitivity       : {sens:.4f}  (recall on MI)")
        print(f"  Specificity       : {spec:.4f}  (recall on NORM)")
        print(f"  Precision         : {prec:.4f}")
        print(f"  F1 Score          : {f1:.4f}")
        print("=" * 50)
        print("  Confusion Matrix")
        print(f"               Pred NORM   Pred MI")
        print(f"  True NORM  :   {tn:>5}      {fp:>5}")
        print(f"  True MI    :   {fn:>5}      {tp:>5}")
        print("=" * 50)
        print(classification_report(all_targets, preds,
                                    target_names=["NORM", "MI"]))

        return {
            "auc": auc, "accuracy": acc, "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1,
            "confusion_matrix": cm, "threshold": threshold
        }