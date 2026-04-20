import torch
import numpy as np
import matplotlib.pyplot as plt
import wfdb


class CAM_FCN:
    def __init__(self, model):
        self.model = model
        self.cam = None

    def _compute(self, x):
        self.model.eval()

        with torch.no_grad():
            fmap = self.model.net(x)
            pooled = self.model.pool(fmap)
            pooled = pooled.squeeze(-1)
            x_out = self.model.head(pooled)

        weights = self.model.head.weight.squeeze(0).unsqueeze(-1)
        fmap = fmap.squeeze(0)

        self.cam = (fmap * weights).sum(dim=0).detach()
        self.cam = torch.relu(self.cam)

        self.cam = (self.cam - self.cam.min()) / (self.cam.max() - self.cam.min())


        return self.cam

    def plot_cam_12leads(self, dataset, idx):
        x = dataset.__getitem__(idx)[0].unsqueeze(0)
        self._compute(x).numpy()

        row = dataset.frame.iloc[idx]
        signal, _ = wfdb.rdsamp(dataset.data_path + row['filename_lr'])

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        label = 'MI' if row['has_mi'] else 'NORM'
        sex = 'Female' if row['sex'] == 1 else 'Male'

        fig, axes = plt.subplots(6, 2, figsize=(16, 14), sharex=True)
        axes = axes.flatten()
        time = np.arange(signal.shape[0])

        for lead_idx, ax in enumerate(axes):
            raw = signal[:, lead_idx]
            ymin = raw.min() - abs(raw.min()) * 0.1
            ymax = raw.max() + abs(raw.max()) * 0.1

            ax.plot(time, raw, color='black', linewidth=0.7, zorder=2)
            ax.imshow(self.cam[np.newaxis, :], aspect='auto', alpha=0.5,
                    extent=[0, len(raw), ymin, ymax],
                    cmap='Reds', vmin=0, vmax=1, zorder=1)

            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(range(1, 13)[lead_idx], fontsize=9, rotation=0, labelpad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.15)

        fig.suptitle(f'{sex}  |  True: {label}  | Predicted prob: {prob:.3f})', fontsize=13)
        plt.tight_layout()
        plt.show()


class CAM_ResNet:
    def __init__(self, model):
        self.model = model
        self.cam = None

    def _compute(self, x):
        self.model.eval()

        with torch.no_grad():
            x = self.model.block1(x)
            x = self.model.drop1(x)
            x = self.model.block2(x)
            x = self.model.drop2(x)
            fmap = self.model.block3(x)
            x = self.model.pool(fmap)
            x = x.squeeze(-1)
            x_out = self.model.head(x)

        weights = self.model.head.weight.squeeze(0).unsqueeze(-1)
        fmap = fmap.squeeze(0)

        self.cam = (fmap * weights).sum(dim=0).detach()
        self.cam = torch.relu(self.cam)

        self.cam = (self.cam - self.cam.min()) / (self.cam.max() - self.cam.min())


        return self.cam

    def plot_cam_12leads(self, dataset, idx):
        x = dataset.__getitem__(idx)[0].unsqueeze(0)
        self._compute(x).numpy()

        row = dataset.frame.iloc[idx]
        signal, _ = wfdb.rdsamp(dataset.data_path + row['filename_lr'])

        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()

        label = 'MI' if row['has_mi'] else 'NORM'
        sex = 'Female' if row['sex'] == 1 else 'Male'

        fig, axes = plt.subplots(6, 2, figsize=(16, 14), sharex=True)
        axes = axes.flatten()
        time = np.arange(signal.shape[0])

        for lead_idx, ax in enumerate(axes):
            raw = signal[:, lead_idx]
            ymin = raw.min() - abs(raw.min()) * 0.1
            ymax = raw.max() + abs(raw.max()) * 0.1

            ax.plot(time, raw, color='black', linewidth=0.7, zorder=2)
            ax.imshow(self.cam[np.newaxis, :], aspect='auto', alpha=0.5,
                    extent=[0, len(raw), ymin, ymax],
                    cmap='Reds', vmin=0, vmax=1, zorder=1)

            ax.set_ylim(ymin, ymax)
            ax.set_ylabel(range(1, 13)[lead_idx], fontsize=9, rotation=0, labelpad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.15)

        fig.suptitle(f'{sex}  |  True: {label}  | Predicted prob: {prob:.3f})', fontsize=20)
        plt.tight_layout()
        plt.show()