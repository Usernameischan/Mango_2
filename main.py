import argparse
# from train_teacher_forcing import *
from train_with_sampling import *
from DataLoader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from helpers import *
from inference import *
from model import Transformer

import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import flwr as fl

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys


def main(
    cid: int = 0,
    epoch: int = 1000,
    k: int = 60,
    batch_size: int = 1,
    frequency: int = 100,
    training_length = 48,
    forecast_window = 24,
    train_csv = r"train_reach_dataset.csv",
    test_csv = r"test_reach_dataset.csv",
    path_to_save_model = r"save_model/",
    path_to_save_loss = r"save_loss/", 
    path_to_save_predictions = r"save_predictions/", 
    device = "cpu"
):

    clean_directory()
    
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer().double().to(device)

    # train_dataset = SensorDataset(csv_name = train_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    train_dataset = load_partition(cid, train_csv, "Data/", training_length, forecast_window)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    # test_dataset = SensorDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    test_dataset = load_partition(cid, test_csv, "Data/", training_length, forecast_window)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Define Flower client
    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            global best_model
            best_model = transformer(model, train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
            return self.get_parameters(config={}), len(train_dataloader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = inference(model, path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)
            return loss, len(test_dataloader.dataset), {"MSE": loss}
        
        
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
    
    
    

    # best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)
    # inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000) ######################### 로컬 에폭 defalut=500, 1000, 2000으로 변경해가면서
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frequency", type=int, default=100)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--path_to_save_loss",type=str,default="save_loss/")
    parser.add_argument("--path_to_save_predictions",type=str,default="save_predictions/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        cid=args.cid,
        epoch=args.epoch,
        k = args.k,
        batch_size=args.batch_size,
        frequency=args.frequency,
        path_to_save_model=args.path_to_save_model,
        path_to_save_loss=args.path_to_save_loss,
        path_to_save_predictions=args.path_to_save_predictions,
        device=args.device,
    )




# net = Net().to(DEVICE)
# trainloaders, valloaders, testloader = load_data()

# cid = int(sys.argv[1])
# trainloader = trainloaders[cid]
# valloader = valloaders[cid]

# # Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         train(net, trainloader, epochs=1)
#         return self.get_parameters(config={}), len(trainloader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss = test(net, testloader)
#         return loss, len(testloader.dataset), {"MSE": loss}


# # Start Flower client
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=FlowerClient(),
# )