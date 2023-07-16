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

import torch 
from torch import nn 
import torch.nn.functional as F
import argparse
import util
import random
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 
import os 
import pandas as pd 
from datetime import date 
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LSTM(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True) 
        
    # 은닉층 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x




def train(model, trainloader, epochs=10, verbose=10, patience=10):
    
    criterion = nn.MSELoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    # epoch마다 loss 저장
    train_hist = np.zeros(epochs)

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = len(trainloader)
        
        for features, labels in enumerate(trainloader):
            
            # seq별 hidden state reset
            model.reset_hidden_state()
            
            # cost 계산
            loss = criterion(net(features).to(DEVICE), labels.to(DEVICE))                    
            
            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_cost += loss/total_batch
        
        train_hist[epoch] = avg_cost        
        # ########################################################################################################
        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):  
            # loss가 커졌다면 early stop
            if train_hist[epoch-patience] < train_hist[epoch]:
                print('\n Early Stopping')
                
                break
            
    return model.eval(), train_hist

    
# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

#     for _ in range(epochs):
#         for features, labels in tqdm(trainloader):
#             optimizer.zero_grad()
#             criterion(net(features.to(DEVICE)), labels.to(DEVICE)).backward()
#             optimizer.step()




def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    # correct, loss = 0, 0.0
    loss, total = 0.0, 0.0
    with torch.no_grad():
        for features, labels in tqdm(testloader):
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item() * len(labels)
            total += len(labels)
    loss /= total
            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    # accuracy = correct / len(testloader.dataset)
    return loss



# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = nn.MSELoss().to(DEVICE) 

#     with torch.no_grad(): 
#         pred = []
#         for pr in range(len(testX_tensor)):

#             model.reset_hidden_state()

#             predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
#             predicted = torch.flatten(predicted).item()
#             pred.append(predicted)

#         # INVERSE
#         pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
#         testY_inverse = scaler_y.inverse_transform(testY_tensor)


# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for features, labels in tqdm(testloader):
#             outputs = net(features.to(DEVICE))
#             labels = labels.to(DEVICE)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple DNN, CNC Dataset)

class reachdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]



def load_data():
    df = pd.read_csv(r"C:\Users\82104\Documents\reach_data222.csv")
    X = df.loc[:, 'Store Number':'Waste quantity']
    y = df['Inventory quantity']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    
    data_length = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값

    X_train = X[:data_length] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
    y_train = y[:data_length] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
    X_test = X[data_length:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
    y_test = y[data_length:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장    

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # X_train = Variable(torch.Tensor(X_train))
    # X_test = Variable(torch.Tensor(X_test))
    
    datasets = []
    trainloaders = []
    valloaders = []

    BATCH_SIZE = 128

    X_test = torch.Tensor(np.array(X_test))
    # print(X_test)
    y_test = np.array(y_test)
    testset = reachdataset(X_test, y_test)
    # print(testset)

    for i in range(3):
        X_cid = X_train[X_train['Store Number'] == i].loc[:, 'Sales amount':]
        X_cid = torch.Tensor(np.array(X))
        y_cid = y_train[X_train['Store Number'] == i]
        y_cid = np.array(y)
        dataset = reachdataset(X_cid, y_cid)
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





if __name__ == "__main__":
    seq_length = 7 # sequence 길이
    data_dim = 7 # Feature 개수
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.001
    epochs = 200
    batch_size = 64
    
    
    net = LSTM(data_dim, hidden_dim, seq_length, output_dim, 1).to(DEVICE)

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
            train(net, trainloader)
            return self.get_parameters(config={}), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = test(net, testloader)
            return loss, len(testloader.dataset)


    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )