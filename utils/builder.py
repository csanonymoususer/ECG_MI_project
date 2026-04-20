import torch

from models.fcn import FCN
from models.resnet import ResNet
from trainers.base_trainer import BaseTrainer


def build_model(config):
    model_name = config["model"]["name"]

    if model_name == "fcn":
        return FCN()

    elif model_name == "resnet":
        return ResNet()

    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

def build_criterion(config):
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(config["training"]["pos_weight"])))

def build_scheduler(optimizer, config):
    if config["training"]["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=float(config["training"]["step_size"]),
            gamma=float(config["training"]["gamma"])
        )
    elif config["training"]["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=float(config["training"]["epochs"]),
            eta_min=float(config["training"]["lr"]) / 10
        )
    return None


def build_trainer(config):
    device = torch.device(config["experiment"]["device"])
    model = build_model(config)
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    criterion = build_criterion(config)
    if config["data"]["input_type"] == "ecg":
        return BaseTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config
        )
