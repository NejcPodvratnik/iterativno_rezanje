import os
import torch
import torchvision.transforms.functional
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np

def getRealWasteDataset():
    dataset_path = "./data/realwaste-main/RealWaste"

    dataset_sub_paths = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]

    inputs, string_targets = [], []
    for path in dataset_sub_paths:
        for root, _, files in os.walk(path):
            for i in range(len(files) - 1):
                inputs += [[os.path.join(root, files[i]), 0]]
                string_targets += [root.split("\\")[-1]]

    inputs, string_targets = np.array(inputs), np.array(string_targets)

    label_encoder = LabelEncoder()
    integer_targets = label_encoder.fit_transform(string_targets)
    integer_targets = integer_targets.reshape(-1, 1)

    permutation = np.random.permutation(len(inputs))
    inputs = inputs[permutation]
    integer_targets = integer_targets[permutation]

    return inputs, integer_targets, label_encoder


class RealWasteDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, image_size, augment) -> None:
        super().__init__()
        self.image_size = image_size
        self.augment = augment
        self.inputs = inputs
        self.targets = targets

        if augment:
            augmented_inputs = []
            augmented_targets = []
            for i, input in enumerate(inputs):
                augmented_inputs += [[input[0], 0]]
                augmented_inputs += [[input[0], 1]]
                augmented_inputs += [[input[0], 2]]
                augmented_targets += [targets[i]]
                augmented_targets += [targets[i]]
                augmented_targets += [targets[i]]

            self.inputs = np.array(augmented_inputs)
            self.targets = np.array(augmented_targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = cv2.imread(self.inputs[index][0])
        target = self.targets[index]

        input = cv2.resize(input, self.image_size)
        
        if self.inputs[index][1] == "1":
            input = cv2.flip(input, 1)
        elif self.inputs[index][1] == "2":
            input = cv2.rotate(input, cv2.ROTATE_90_CLOCKWISE)

        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        input = torch.from_numpy(input).permute(2, 0, 1)
        target = torch.from_numpy(target).squeeze()
        
        input = input.type(torch.float32) / 255.
        #input = torchvision.transforms.functional.normalize(input, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        return input, target