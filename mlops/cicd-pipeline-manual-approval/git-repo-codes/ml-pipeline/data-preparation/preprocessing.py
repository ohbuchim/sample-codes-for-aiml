
import argparse
# import boto3
import os
import tarfile
import warnings
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default='/opt/ml/processing/input')
    parser.add_argument("--output-dir", type=str, default='/opt/ml/processing/output')
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    filename = os.path.join(args.input_dir, 'mnist_png.tgz')

    with tarfile.open(filename, 'r:gz') as t:
        t.extractall(path='mnist')

    prepared_data_path = args.output_dir
    os.makedirs(prepared_data_path, exist_ok=True)

    training_dir = 'mnist/mnist_png/training'
    test_dir = 'mnist/mnist_png/testing'

    training_data = datasets.ImageFolder(
        root=training_dir,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]))

    training_data_loader = DataLoader(training_data,
                                      batch_size=len(training_data))
    training_data_loaded = next(iter(training_data_loader))
    torch.save(training_data_loaded,
               os.path.join(prepared_data_path, 'training.pt'))

    test_data_loader = DataLoader(test_data, batch_size=len(test_data))
    test_data_loaded = next(iter(test_data_loader))
    torch.save(test_data_loaded,
               os.path.join(prepared_data_path, 'test.pt'))
