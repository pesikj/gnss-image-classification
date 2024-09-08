from __future__ import print_function, division

import copy
import csv
import itertools
import os
import shutil
import sys
import time
from dataclasses import asdict
from typing import List

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

from run_statistics import RunStatistics
from preprocess_images import GROUP_COUNT

PROCESSED_DATA_DIR = 'data_processed'
ACTIVE_DATA_DIR = 'data_active_dir'
STATISTICS_FILENAME = "test_statistics.csv"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_model(model):
    match model:
        case "resnet18":
            return models.resnet18(pretrained=True)
        case "resnet34":
            return models.resnet34(pretrained=False)
        case "resnet50":
            return models.resnet50(pretrained=True)
        case "resnet101":
            return models.resnet101(pretrained=True)
        case "resnet152":
            return models.resnet152(pretrained=True)
        case _:
            Exception(f"Unknown model: {model}")


def get_data(current_val_group):
    data_transforms = {
        'train': transforms.Compose([
            # https://www.geeksforgeeks.org/randomresizedcrop-method-in-python-pytorch/
            # https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
            # transforms.RandomResizedCrop(224),
            # https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html
            # transforms.RandomHorizontalFlip(),
            # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
            # transforms.Resize(256),
            transforms.CenterCrop([790, 340]),
            transforms.ToTensor(),
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop([790, 340]),
            # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
                if not os.path.exists(destination):
                    os.makedirs(destination)
                shutil.copy2(os.path.join(PROCESSED_DATA_DIR, str(i), category, file), os.path.join(destination, file))

    image_datasets = {x: datasets.ImageFolder(os.path.join(ACTIVE_DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=5,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, class_names


def load_statistics():
    statistics = []
    if os.path.exists(STATISTICS_FILENAME):
        with open(STATISTICS_FILENAME) as file:
            reader = csv.DictReader(file)
            for row in reader:
                row: dict
                statistics.append(
                    RunStatistics(cross_validation_iteration=row['cross_validation_iteration'],
                                  model=row["model"],
                                  epoch=row["epoch"],
                                  epoch_total=row["epoch_total"],
                                  phase=row["phase"],
                                  epoch_loss=row["epoch_loss"],
                                  epoch_acc=row["epoch_acc"])
                )
    return statistics


def check_if_complete(model, statistics, num_epochs):
    match_items = [x for x in statistics if x.model == model]
    return len(match_items) == num_epochs * 2 * GROUP_COUNT


def save_statistics(statistics):
    dict_list = [asdict(stat) for stat in statistics]
    fieldnames = dict_list[0].keys() if dict_list else []
    with open(STATISTICS_FILENAME, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_list)


def run_image_classification(model_name, iteration, statistics, num_epochs):
    cudnn.benchmark = True
    plt.ion()
    dataloaders, dataset_sizes, class_names = get_data(iteration)

    def imshow(inp, title=None, filename=None, path=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        if filename is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, f"{filename}.png"))

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    def train_model(model, criterion, optimizer, scheduler):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]

                statistics.append(
                    RunStatistics(iteration, model_name, epoch + 1, num_epochs, phase, epoch_loss, epoch_acc.item()))
                save_statistics(statistics)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    confusion_matrix = numpy.zeros((len(class_names), len(class_names)))
    confusion_matrix = pandas.DataFrame(confusion_matrix, index=class_names, columns=class_names)

    def visualize_model(model, num_images=6):
        print("Model visualization")
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    correct_class = class_names[labels[j]]
                    predicted_class = class_names[preds[j]]
                    confusion_matrix.loc[correct_class, predicted_class] += 1
                    path = os.path.join("output", correct_class,
                                        "Správně zařazené" if predicted_class == correct_class else predicted_class
                                        )
                    title = correct_class if predicted_class == correct_class \
                        else f"{correct_class}, predicted as {predicted_class}"
                    imshow(inputs.cpu().data[j], title=title, filename=f"{i}_{j}", path=path)

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    model_ft = get_model(model_name)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
    visualize_model(model_ft)
    confusion_matrix.to_excel(os.path.join("confusion_matrix_output", f"{model_name}_{iteration}.xlsx"))


def main(num_epochs, models):
    statistics = load_statistics()
    statistics_new = []
    iterations = range(1, GROUP_COUNT + 1)
    for model in models:
        model_complete = check_if_complete(model, statistics, num_epochs)
        if model_complete:
            for item in statistics:
                if item.model == model:
                    statistics_new.append(item)
    statistics = statistics_new
    save_statistics(statistics)
    for model, i in itertools.product(models, iterations):
        model_complete = check_if_complete(model, statistics, num_epochs)
        if not model_complete:
            run_image_classification(model, i, statistics, num_epochs)


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2:])
