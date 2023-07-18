import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_dim=49, hidden_dim=20, output_dim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    for _ in range(epochs):
        for features, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(features.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()



def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for features, labels in tqdm(testloader):
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy



# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple DNN, CNC Dataset)

class CNCdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]


def load_data():
    df = pd.read_csv(r"C:\Users\uoon9\GitHub\Mango_2\data\03. Dataset_CNC\preprocess\CNC.csv")
    X = df.loc[:, ~df.columns.isin(['target'])]
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    datasets = []
    trainloaders = []
    valloaders = []

    BATCH_SIZE = 32

    X_test = torch.Tensor(np.array(X_test))
    y_test = np.array(y_test)
    testset = CNCdataset(X_test, y_test)

    for i in range(2):
        X_cid = X_train[X_train.cid == i].loc[:, ~X_train.columns.isin(['cid'])]
        X_cid = torch.Tensor(np.array(X))
        y_cid = y_train[X_train.cid == i]
        y_cid = np.array(y)
        dataset = CNCdataset(X_cid, y_cid)
        datasets.append(dataset)

    for dataset in datasets:
        len_val = len(dataset) // 10  # 10 % validation set
        len_train = len(dataset) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size = BATCH_SIZE, shuffle = True))
        valloaders.append(DataLoader(ds_val, batch_size = BATCH_SIZE))
    testloader = DataLoader(testset, batch_size = BATCH_SIZE)

    return trainloaders, valloaders, testloader


net = Net().to(DEVICE)
trainloaders, valloaders, testloader = load_data()

cid = int(sys.argv[1])
trainloader = trainloaders[cid]
valloader = valloaders[cid]

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)