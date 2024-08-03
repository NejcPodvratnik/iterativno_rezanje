import os
import torch
import torchvision.transforms.functional
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import pandas as pd


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, inputs_folder, targets_folder, img_size) -> None:
        super().__init__()

        self.inputs_path = []
        for file in sorted(os.listdir(inputs_folder)):
            self.inputs_path += [f"{inputs_folder}/{file}"]

        self.targets = pd.read_csv(targets_folder)
        self.targets = list(self.targets.sort_values(by='imageId')['age'])

        self.inputs_path = np.array(self.inputs_path)
        self.targets = np.array(self.targets)

        self.img_size = img_size

    def __len__(self):
        return len(self.inputs_path)

    def __getitem__(self, index):
        input = cv2.imread(self.inputs_path[index])
        target = self.targets[index]

        input = 255 - input
        input = cv2.resize(input, self.img_size)

        input = np.array(input)
        input = input.transpose((2,0,1))
        input = torch.from_numpy(input)
        target = torch.from_numpy(np.array([target])).float()
        
        input = input.type(torch.float32) / 255.
        #input = torchvision.transforms.functional.normalize(input, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        return input, target