## Functions for loading the CIFAR-10 dataset and creating DataLoaders

import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np

def load_transformed_dataset(IMG_SIZE = 32, BATCH_SIZE=128, train = True):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)
    train = None
    if(train):
        train = torchvision.datasets.CIFAR10(root="./datasets/cifar10", download=True, 
                                         transform=data_transform)
    if(not train):
        train = torchvision.datasets.CIFAR10(root="./datasets/cifar10", download=True, 
                                         transform=data_transform, train=False)
    return train



## To get dataset and Dataloader
def get_dataset_and_dataloader(IMG_SIZE=32, BATCH_SIZE=128, train = True):
    data = load_transformed_dataset(IMG_SIZE, BATCH_SIZE, train)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return dataloader




