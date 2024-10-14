import sys
import torch
from torch.utils.data import DataLoader

from dataset_realwaste import *
from dataset_chest_xray import *
from iterative_pruning import IterativePruning

from torchvision.models import densenet121
from torchvision.models import resnet18
from torchvision.models import alexnet
from torchvision.models import vgg11
from torchvision.models import shufflenet_v2_x1_0

NUM_OF_RUNS = 5
PATIENCE = 5

#[velikost za realwaste dataset, velikost za xray dataset]
RESNET18_BATCH_SIZE = [128, 128]
ALEXNET_BATCH_SIZE = [128, 128]
VGG11_BATCH_SIZE = [128, 128]

XRAY_IMAGE_SIZE = [256, 256]
REALWASTE_IMAGE_SIZE = [524, 524]

MODELS = ["vgg11", "alexnet", "resnet18"]
DATASETS = ["realwaste", "xray"]

PRUNE_PERCENTAGES = [0.25, 0.5, 0.75]
PRUNE_PERCENTAGES_ITERATIONS = [18, 10, 6]

#[lr za realwaste dataset, lr za xray dataset]
RESNET18_LEARNING_RATES = [0.0001, 0.001]
ALEXNET_LEARNING_RATES = [0.0001, 0.0001]
VGG11_LEARNING_RATES = [0.0001, 0.0003]

USE_PRETRAINED_WEIGHTS = False

def run(test_name, model_name, dataset_name, num_of_runs, batch_size, image_size, num_max_epochs, num_prume_iterations, prune_percentage, learning_rate, use_pretrained_weights):
    if not os.path.exists("./tests"):
        os.makedirs("./tests")

    model_filepath = f"./tests/{test_name}"
    os.makedirs(model_filepath)
    for i in range(num_of_runs):
        if dataset_name == "xray":
            chest_xray_dataset = ChestXRayDataset(image_size)

            train_size = int(0.7 * len(chest_xray_dataset))
            val_size = int(0.15 * len(chest_xray_dataset))
            test_size = len(chest_xray_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(chest_xray_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1))
        elif dataset_name == "realwaste":
            inputs, targets, _ = getRealWasteDataset()

            train_size = int(0.7 * len(inputs))
            val_size = int(0.15 * len(inputs))
            test_size = len(inputs) - train_size - val_size

            train_dataset = RealWasteDataset(inputs[:train_size,:], targets[:train_size], image_size, True)
            val_dataset = RealWasteDataset(inputs[train_size : train_size + val_size,:], targets[train_size : train_size + val_size], image_size, False)
            test_dataset = RealWasteDataset(inputs[-test_size:,:], targets[-test_size:], image_size, False)
        else:
            print("Wrong dataset used.")
            return
        

        train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
        val_loader = DataLoader(dataset = val_dataset, shuffle = False, batch_size = batch_size)
        test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = batch_size)


        num_channels = 1 if dataset_name == "xray" else 10
        weights = "IMAGENET1K_V1" if use_pretrained_weights else None
        if model_name == "resnet18":
            model = resnet18(weights = weights)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_channels)
        elif model_name == "alexnet":
            model = alexnet(weights = weights)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_channels)
        elif model_name == "vgg11":
            model = vgg11(weights = weights)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_channels)
        elif model_name == "shuffle_net":
            model = shufflenet_v2_x1_0(weights = weights)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_channels)
        elif model_name == "densenet121":
            model = densenet121(wweights = weights)
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, num_channels)
        else:
            print("Wrong model used.")
            return

        if dataset_name == "xray":
            loss_fn = torch.nn.MSELoss()
        elif dataset_name == "realwaste":
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            print("Wrong dataset used.")
            return

        ip = IterativePruning(model_name, model, apply_weights = not use_pretrained_weights)
        ip.start(i, num_of_runs, model_filepath + f"/test_{i + 1}.json", loss_fn, train_loader, val_loader, test_loader, learning_rate, num_max_epochs, num_prume_iterations, prune_percentage, patience = PATIENCE)

if __name__ == "__main__":

    batch_sizes = [VGG11_BATCH_SIZE, ALEXNET_BATCH_SIZE, RESNET18_BATCH_SIZE]
    image_sizes = [REALWASTE_IMAGE_SIZE, XRAY_IMAGE_SIZE]
    learning_rates = [VGG11_LEARNING_RATES, ALEXNET_LEARNING_RATES, RESNET18_LEARNING_RATES]


    for m, model in enumerate(MODELS):
        for d, dataset in enumerate(DATASETS):
            for i in range(len(PRUNE_PERCENTAGES)):
                print(f"{model} {dataset} {batch_sizes[m][d]} {image_sizes[d]} {PRUNE_PERCENTAGES_ITERATIONS[i]} {PRUNE_PERCENTAGES[i]} {learning_rates[m][d]}")
                run(f"{model}_{dataset}_p_{PRUNE_PERCENTAGES[i]}", model, dataset, NUM_OF_RUNS, batch_sizes[m][d], image_sizes[d], 100, PRUNE_PERCENTAGES_ITERATIONS[i], PRUNE_PERCENTAGES[i], learning_rates[m][d], USE_PRETRAINED_WEIGHTS)