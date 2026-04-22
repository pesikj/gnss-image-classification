import copy
import csv
import itertools
import os
import shutil
import sys
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from run_statistics import RunStatistics
from preprocess_images import GROUP_COUNT

PROCESSED_DATA_DIR = "data_processed"
ACTIVE_DATA_DIR = "data_active_dir"
CONFUSION_MATRIX_DIR = "confusion_matrix_output"
STATISTICS_FILENAME = "test_statistics.csv"

IMAGE_CROP = (790, 340)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 5
NUM_WORKERS = 4
LR = 0.001
MOMENTUM = 0.9
LR_STEP_SIZE = 7
LR_GAMMA = 0.1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_model(model_name: str) -> nn.Module:
    match model_name:
        case "resnet18":
            return models.resnet18(weights=ResNet18_Weights.DEFAULT)
        case "resnet34":
            return models.resnet34(weights=None)
        case "resnet50":
            return models.resnet50(weights=ResNet50_Weights.DEFAULT)
        case "resnet101":
            return models.resnet101(weights=ResNet101_Weights.DEFAULT)
        case "resnet152":
            return models.resnet152(weights=ResNet152_Weights.DEFAULT)
        case _:
            raise ValueError(f"Unknown model: {model_name!r}")


def get_data(current_val_group: int) -> tuple[dict[str, DataLoader], dict[str, int], list[str]]:
    data_transforms = {
        "train": transforms.Compose([
            transforms.CenterCrop(IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "val": transforms.Compose([
            transforms.CenterCrop(IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }

    if os.path.exists(ACTIVE_DATA_DIR):
        shutil.rmtree(ACTIVE_DATA_DIR)
    os.makedirs(os.path.join(ACTIVE_DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(ACTIVE_DATA_DIR, "val"), exist_ok=True)

    for i in range(1, GROUP_COUNT + 1):
        target_folder = os.path.join(ACTIVE_DATA_DIR, "val" if i == current_val_group else "train")
        for category in os.listdir(os.path.join(PROCESSED_DATA_DIR, str(i))):
            for file in os.listdir(os.path.join(PROCESSED_DATA_DIR, str(i), category)):
                destination = os.path.join(target_folder, category)
                os.makedirs(destination, exist_ok=True)
                shutil.copy2(
                    os.path.join(PROCESSED_DATA_DIR, str(i), category, file),
                    os.path.join(destination, file),
                )

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(ACTIVE_DATA_DIR, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


def save_prediction_image(
    tensor: torch.Tensor,
    title: str | None,
    filename: str,
    path: str,
) -> None:
    inp = tensor.numpy().transpose((1, 2, 0))
    inp = np.array(IMAGENET_STD) * inp + np.array(IMAGENET_MEAN)
    inp = np.clip(inp, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(inp)
    if title is not None:
        ax.set_title(title)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{filename}.png"))
    plt.close(fig)


def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.StepLR,
    dataloaders: dict[str, DataLoader],
    dataset_sizes: dict[str, int],
    model_name: str,
    iteration: int,
    statistics: list[RunStatistics],
    num_epochs: int,
) -> nn.Module:
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            statistics.append(
                RunStatistics(iteration, model_name, epoch + 1, num_epochs, phase, epoch_loss, epoch_acc.item())
            )
            save_statistics(statistics)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    model.load_state_dict(best_model_weights)
    return model


def evaluate_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    class_names: list[str],
) -> pd.DataFrame:
    conf_matrix = pd.DataFrame(
        np.zeros((len(class_names), len(class_names))),
        index=class_names,
        columns=class_names,
    )

    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                true_class = class_names[labels[j]]
                pred_class = class_names[preds[j]]
                conf_matrix.loc[true_class, pred_class] += 1
                correctly_classified = pred_class == true_class
                path = os.path.join(
                    "output", true_class,
                    "Správně zařazené" if correctly_classified else pred_class,
                )
                title = true_class if correctly_classified else f"{true_class}, predicted as {pred_class}"
                save_prediction_image(inputs.cpu().data[j], title=title, filename=f"{i}_{j}", path=path)

    return conf_matrix


def load_statistics() -> list[RunStatistics]:
    if not os.path.exists(STATISTICS_FILENAME):
        return []
    statistics = []
    with open(STATISTICS_FILENAME) as file:
        reader = csv.DictReader(file)
        for row in reader:
            statistics.append(
                RunStatistics(
                    cross_validation_iteration=row["cross_validation_iteration"],
                    model=row["model"],
                    epoch=row["epoch"],
                    epoch_total=row["epoch_total"],
                    phase=row["phase"],
                    epoch_loss=row["epoch_loss"],
                    epoch_acc=row["epoch_acc"],
                )
            )
    return statistics


def save_statistics(statistics: list[RunStatistics]) -> None:
    if not statistics:
        return
    dict_list = [asdict(stat) for stat in statistics]
    fieldnames = dict_list[0].keys()
    with open(STATISTICS_FILENAME, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_list)


def check_if_complete(model_name: str, statistics: list[RunStatistics], num_epochs: int) -> bool:
    matching = [s for s in statistics if s.model == model_name]
    return len(matching) == num_epochs * 2 * GROUP_COUNT


def run_image_classification(
    model_name: str,
    iteration: int,
    statistics: list[RunStatistics],
    num_epochs: int,
) -> None:
    cudnn.benchmark = True
    dataloaders, dataset_sizes, class_names = get_data(iteration)

    model = get_model(model_name)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    model = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, model_name, iteration, statistics, num_epochs,
    )

    conf_matrix = evaluate_model(model, dataloaders, class_names)
    os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
    conf_matrix.to_excel(os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_{iteration}.xlsx"))


def main(num_epochs: int, model_names: list[str]) -> None:
    statistics = load_statistics()

    complete = {m for m in model_names if check_if_complete(m, statistics, num_epochs)}
    statistics = [s for s in statistics if s.model in complete]
    save_statistics(statistics)

    for model_name, iteration in itertools.product(model_names, range(1, GROUP_COUNT + 1)):
        if not check_if_complete(model_name, statistics, num_epochs):
            run_image_classification(model_name, iteration, statistics, num_epochs)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2:])
