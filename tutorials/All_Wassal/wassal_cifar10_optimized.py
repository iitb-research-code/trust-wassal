import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Any

# Assuming model and strategy classes are defined in respective modules
from models import ResNet18, ResNet50
from strategies import BADGE, EntropySampling, GLISTER, GradMatchActive, CoreSet, LeastConfidenceSampling, MarginSampling, RandomSampling
from utils import load_dataset_custom, LabeledToUnlabeledDataset, SubsetWithTargets, ConcatWithTargets

@dataclass
class ExperimentConfig:
    datadir: str = "data"
    dataset_name: str = "cifar10"
    feature: str = "classimb"
    model_name: str = "ResNet18"
    budget: int = 50
    learning_rate: float = 0.0003
    num_rounds: int = 10
    split_cfg: dict = None
    device: str = "cpu"
    compute_error_log: bool = True
    strategy: str = "SIM"
    method: str = ""
    embedding_type: str = "features"
    soft_loss_hyperparam: float = 3.0

def initialize_config():
    split_cfg = {
        "num_cls_imbalance": list(range(10)),
        "sel_cls_idx": list(range(10)),
        "per_class_train": [20] * 10,
        "per_class_val": [10] * 10,
        "per_class_lake": [400] * 10,
        "per_imbclass_train": [20] * 10,
        "per_imbclass_val": [10] * 10,
        "per_imbclass_lake": [400] * 10
    }
    return ExperimentConfig(split_cfg=split_cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")

def setup_directories(experiment_name: str, method: str, budget: int, run: str) -> str:
    base_dir = Path(f"/home/venkatapathy/trust-wassal/tutorials/results/{experiment_name}")
    result_dir = base_dir / method / str(budget) / run
    result_dir.mkdir(parents=True, exist_ok=True)
    return str(result_dir)

def load_model(config: ExperimentConfig, num_classes: int):
    if config.model_name == "ResNet18":
        model = ResNet18(num_classes)
    elif config.model_name == "ResNet50":
        model = ResNet50(num_classes)
    else:
        raise ValueError(f"Unsupported model type: {config.model_name}")
    model.to(config.device)
    return model

def setup_data_loaders(config: ExperimentConfig):
    train_set, val_set, test_set, lake_set = load_dataset_custom(
        config.datadir, config.dataset_name, config.feature, config.split_cfg
    )
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=200, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=False)
    return train_loader, val_loader, test_loader, lake_set

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
    accuracy = correct / len(data_loader.dataset)
    return total_loss, accuracy

def initialize_strategy(config: ExperimentConfig, train_set, unlabeled_set, model):
    if config.strategy == "AL" and config.method == "badge":
        strategy = BADGE(train_set, unlabeled_set, model, num_classes=10)
    # Add other strategies as elif blocks here
    return strategy

def run_experiment(config: ExperimentConfig):
    result_dir = setup_directories("experiment_name", config.method, config.budget, "run_identifier")
    model = load_model(config, 10)  # Assuming 10 classes
    train_loader, val_loader, test_loader, lake_set = setup_data_loaders(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    strategy = initialize_strategy(config, train_set, unlabeled_set, model)

    for epoch in range(config.num_rounds):
        train_loss = train_model(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, config.device)

        selected_indices = strategy.select(config.budget)
        update_training_set(train_set, selected_indices)

        print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        torch.save(model.state_dict(), Path(result_dir) / f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    config = initialize_config()
    run_experiment(config)
