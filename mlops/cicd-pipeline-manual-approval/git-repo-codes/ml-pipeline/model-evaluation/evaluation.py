
import json
import os
import tarfile
import argparse

import pandas as pd

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from smexperiments.tracker import Tracker
from smexperiments.trial import Trial
from sagemaker.analytics import ExperimentAnalytics
import sagemaker

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _get_test_data_loader(test_batch_size, training_dir):
    print("Get test data loader")
    test_tensor = torch.load(os.path.join(training_dir, 'test.pt'))
    dataset = torch.utils.data.TensorDataset(test_tensor[0], test_tensor[1])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    average_loss = test_loss
    accuracy = 100. * correct / len(test_loader.dataset)
    return average_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model path')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Data path')
    parser.add_argument('--test-batch-size', type=str, default='8',
                        help='Batch size for inference')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')

    args = parser.parse_args()
    
    model_dir = args.model_dir
    data_dir = args.data_dir
    model_path = os.path.join(model_dir, "model.tar.gz")

    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=".")
    print("Loading model")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Net())

    with open('model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.to(device)

    print("Loading test input data")
    test_loader = _get_test_data_loader(int(args.test_batch_size),
                                        data_dir)

    average_loss, accuracy = test(model, test_loader, device)

    print("Creating evaluation report")
    report_dict = {
        "custom_metrics": {
            "average_loss": {
                "value": average_loss,
                "standard_deviation": 0
            },
            "accuracy": {
                "value": accuracy,
                "standard_deviation": 0
            }
        }
    }

    print(args.experiment_name)
    trial_component_analytics = ExperimentAnalytics(
        experiment_name=args.experiment_name,
        sort_by="parameters.accuracy",
        sort_order="Descending",# Ascending or Descending
    )
    
    df = trial_component_analytics.dataframe()
    is_best = 0
    try:
        best_acc = df.iloc[0]['accuracy'] 
        if best_acc < report_dict["custom_metrics"]["accuracy"]:
            print('This model is the best ever!!')
            is_best = 1
        else:
            print('This model is not so good.')
    except:
        is_best = 1
        print('This model is the first one.')
    
    print('Recording metrics to Experiments...')
    with Tracker.load() as processing_tracker: # Tracker requires with keyword
        processing_tracker.log_parameters({ "accuracy": report_dict["custom_metrics"]["accuracy"], 
                                            "average_loss": report_dict["custom_metrics"]["average_loss"], 
                                            "is_best": is_best})

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving evaluation report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
