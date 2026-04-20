import yaml
import torch
import numpy as np
from utils.builder import build_trainer
from dataset.dataset import Dataset
from utils.config import load_config


CONFIG_PATH = "configs/resnet_full.yaml"


def main(config_path):

    config = load_config(config_path)
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])
    # device = torch.device(config["experiment"]["device"])
    # print(device)

    dataset = Dataset(config["data"]["data_path"],
                      config["data"]["use_data"],
                      config["data"]["batch_size"],
                      use_tabular=config["data"]["use_tabular"],
                      tabular_features=config["data"]["tabular_features"])
    
    train_loader, val_loader, test_loader = dataset.get_loaders()

    trainer = build_trainer(config)
    trainer.load_checkpoint()
    trainer.choose_threshold(val_loader)


if __name__ == "__main__":
    main(CONFIG_PATH)
