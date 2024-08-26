import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def import_MNIST_dataset(mini_data=False):
    """
    Downloads the MNIST dataset and loads it into DataLoader objects for training and testing.

    The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits.
    The images are normalized to have pixel values between -1 and 1.

    :return: A tuple containing the training DataLoader and the testing DataLoader.
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the testing dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    if mini_data:
            indices = torch.arange(100)
            mini_train_set = torch.utils.data.Subset(trainset,indices)
            trainloader = torch.utils.data.DataLoader(mini_train_set, batch_size=16, shuffle=False)
            return trainloader, testloader
    return trainloader, testloader

