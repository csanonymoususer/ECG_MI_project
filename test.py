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
    
    # print(device)

    dataset = Dataset(config["data"]["data_path"],
                      config["data"]["use_data"],
                      config["data"]["batch_size"],
                      use_tabular=config["data"]["use_tabular"],
                      tabular_features=config["data"]["tabular_features"])
    
    train_loader, val_loader, test_loader = dataset.get_loaders()

    
    trainer = build_trainer(config)
    trainer.load_checkpoint()

    # find threshold on full val set
    best_thr = trainer.choose_threshold(val_loader)

    print("\n===== ALL =====")
    trainer.test(test_loader, best_thr)

    dataset_f = Dataset(config["data"]["data_path"], "F",
                        config["data"]["batch_size"])
    _, val_loader_f, test_loader_f = dataset_f.get_loaders()
    print("\n===== FEMALE =====")
    best_thr = trainer.choose_threshold(val_loader_f)
    trainer.test(test_loader_f, best_thr)

    dataset_m = Dataset(config["data"]["data_path"], "M",
                        config["data"]["batch_size"])       
    _, val_loader_m, test_loader_m = dataset_m.get_loaders()
    best_thr = trainer.choose_threshold(val_loader_m)
    print("\n===== MALE =====")
    trainer.test(test_loader_m, best_thr)


if __name__ == "__main__":
    main(CONFIG_PATH)
