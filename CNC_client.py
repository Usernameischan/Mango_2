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
# import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 
import os 
import pandas as pd 
from datetime import date 



# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self, input_dim=49, hidden_dim=20, output_dim=2):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out



class TPALSTM(nn.Module):

    def __init__(self, input_size=5, output_horizon=1, hidden_size=24, obs_len=5000, n_layers=3):
        super(TPALSTM, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, \
                    bias=True, batch_first=True) # output (batch_size, obs_len, hidden_size)
        self.hidden_size = hidden_size
        self.filter_num = 32
        self.filter_size = 1
        self.output_horizon = output_horizon
        self.attention = TemporalPatternAttention(self.filter_size, \
            self.filter_num, obs_len-1, hidden_size)
        self.linear = nn.Linear(hidden_size, output_horizon)
        self.n_layers = n_layers

    def forward(self, x):
        batch_size, obs_len = x.size()
        x = x.view(batch_size, obs_len, 1)
        xconcat = self.relu(self.hidden(x))
        # x = xconcat[:, :obs_len, :]
        # xf = xconcat[:, obs_len:, :]
        H = torch.zeros(batch_size, obs_len-1, self.hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        ct = ht.clone()
        for t in range(obs_len):
            xt = xconcat[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)
        
        # reshape hidden states H
        H = H.view(-1, 1, obs_len-1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht)
        return ypred

class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()
    
    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht) # batch_size, 1, filter_num 
        conv_vecs = self.conv(H)
        
        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)
        
        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht
    
    
# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

#     for _ in range(epochs):
#         for features, labels in tqdm(trainloader):
#             optimizer.zero_grad()
#             criterion(net(features.to(DEVICE)), labels.to(DEVICE)).backward()
#             optimizer.step()

def train(net, trainloader, epochs):
    '''
    Args:
    - X (array like): shape (num_samples, num_features, num_periods)
    - y (array like): shape (num_samples, num_periods)
    - epoches (int): number of epoches to run
    - step_per_epoch (int): steps per epoch to run
    - seq_len (int): output horizon
    - likelihood (str): what type of likelihood to use, default is gaussian
    - num_skus_to_show (int): how many skus to show in test phase
    - num_results_to_sample (int): how many samples in test phase as prediction
    '''
    # num_ts, num_periods, num_features = X.shape
    # model = TPALSTM(1, args.seq_len, 
    #     args.hidden_size, args.num_obs_to_train, args.n_layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    random.seed(2)
    # select sku with most top n quantities 
    # Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    losses = []
    # cnt = 0

    # yscaler = None
    # if args.standard_scaler:
    #     yscaler = util.StandardScaler()
    # elif args.log_scaler:
    #     yscaler = util.LogScaler()
    # elif args.mean_scaler:
    #     yscaler = util.MeanScaler()
    # elif args.max_scaler:
    #     yscaler = util.MaxScaler()
    # if yscaler is not None:
    #     ytr = yscaler.fit_transform(ytr)

    # training
    # seq_len = args.seq_len
    # obs_len = args.num_obs_to_train
    
    criterion = torch.nn.MSELoss()
        
    for epoch in range(epochs):
        # print("Epoch {} starts...".format(epoch))
        for features, labels in tqdm(trainloader):
            
            # loss = util.RSE(ypred, yf)
            criterion(net(features.to(DEVICE)), labels.to(DEVICE))

            losses.append(criterion.item())
            optimizer.zero_grad()
            criterion.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    # correct, loss = 0, 0.0
    loss = 0.0
    with torch.no_grad():
        for features, labels in tqdm(testloader):
            outputs = net(features.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    # accuracy = correct / len(testloader.dataset)
    return loss



# def test(net, testloader, args):

#     num_ts, num_periods, num_features = X.shape
#     model = TPALSTM(1, args.seq_len, 
#         args.hidden_size, args.num_obs_to_train, args.n_layers)

    # select sku with most top n quantities 
    # Xtr, ytr, Xte, yte = util.train_test_split(X, y)
    # losses = []
    # cnt = 0
        
    # test 
    # mape_list = []
    # select skus with most top K
    # X_test = Xte[:, -seq_len-obs_len:-seq_len, :].reshape((num_ts, -1, num_features))
    # Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
    # y_test = yte[:, -seq_len-obs_len:-seq_len].reshape((num_ts, -1))
    # yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
    # yscaler = None  
    # if args.standard_scaler:
    #     yscaler = util.StandardScaler()
    # elif args.log_scaler:
    #     yscaler = util.LogScaler()
    # elif args.mean_scaler:
    #     yscaler = util.MeanScaler()
    # elif args.max_scaler:
    #     yscaler = util.MaxScaler()
    # if yscaler is not None:
    #     ytr = yscaler.fit_transform(ytr)
    # if yscaler is not None:
    #     y_test = yscaler.fit_transform(y_test)
        
    # X_test = torch.from_numpy(X_test).float()
    # y_test = torch.from_numpy(y_test).float()
    # Xf_test = torch.from_numpy(Xf_test).float()
    # ypred = model(y_test)
    # ypred = ypred.data.numpy()
    # if yscaler is not None:
    #     ypred = yscaler.inverse_transform(ypred)
    # ypred = ypred.ravel()
    
    # loss = np.sqrt(np.sum(np.square(yf_test - ypred)))
    # print("losses: ", loss)
    # mape_list.append(loss)

    # if args.show_plot:
    #     plt.figure(1, figsize=(20, 5))
    #     plt.plot([k + seq_len + obs_len - seq_len \
    #         for k in range(seq_len)], ypred, "r-")
    #     plt.title('Prediction uncertainty')
    #     yplot = yte[-1, -seq_len-obs_len:]
    #     plt.plot(range(len(yplot)), yplot, "k-")
    #     plt.legend(["prediction", "true", "P10-P90 quantile"], loc="upper left")
    #     ymin, ymax = plt.ylim()
    #     plt.vlines(seq_len + obs_len - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
    #     plt.ylim(ymin, ymax)
    #     plt.xlabel("Periods")
    #     plt.ylabel("Y")
    #     plt.show()
    # return losses, mape_list

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
    
    datasets = []
    trainloaders = []
    valloaders = []

    BATCH_SIZE = 32

    X_test = torch.Tensor(np.array(X_test))
    y_test = np.array(y_test)
    testset = reachdataset(X_test, y_test)

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
    
    net = TPALSTM().to(DEVICE)
    trainloaders, valloaders, testloader = load_data()

    cid = int(sys.argv[1])
    trainloader = trainloaders[cid]
    valloader = valloaders[cid]

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num_epoches", "-e", type=int, default=1) # 원래는 1000으로 설정되어 있었음.
    # parser.add_argument("--step_per_epoch", "-spe", type=int, default=2)
    # parser.add_argument("-lr", type=float, default=1e-3)
    # parser.add_argument("--n_layers", "-nl", type=int, default=3)
    # parser.add_argument("--hidden_size", "-hs", type=int, default=24)
    # parser.add_argument("--seq_len", "-sl", type=int, default=7)
    # parser.add_argument("--num_obs_to_train", "-not", type=int, default=1)
    # parser.add_argument("--num_results_to_sample", "-nrs", type=int, default=10)
    # parser.add_argument("--show_plot", "-sp", action="store_true")
    # parser.add_argument("--run_test", "-rt", action="store_true")
    # parser.add_argument("--standard_scaler", "-ss", action="store_true")
    # parser.add_argument("--log_scaler", "-ls", action="store_true")
    # parser.add_argument("--mean_scaler", "-ms", action="store_true")
    # parser.add_argument("--max_scaler", "-max", action="store_true")
    # parser.add_argument("--batch_size", "-b", type=int, default=64)
    # parser.add_argument("--sample_size", type=int, default=100)
    
    # args = parser.parse_args()


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
            loss = test(net, testloader)
            return loss, len(testloader.dataset)


    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(),
    )
    
    # if args.run_test:

        # data_path = util.get_data_path()
        # data = pd.read_csv(os.path.join(data_path, "LD_MT200_hour.csv"), parse_dates=["date"])
        # data["year"] = data["date"].apply(lambda x: x.year)
        # data["day_of_week"] = data["date"].apply(lambda x: x.dayofweek)
        # data = data.loc[(data["date"] >= date(2014, 1, 1)) & (data["date"] <= date(2014, 3, 1))]

        # features = ["hour", "day_of_week"]
        # # hours = pd.get_dummies(data["hour"])
        # # dows = pd.get_dummies(data["day_of_week"])
        # hours = data["hour"]
        # dows = data["day_of_week"]
        # X = np.c_[np.asarray(hours), np.asarray(dows)]
        # num_features = X.shape[1]
        # num_periods = len(data)
        # X = np.asarray(X).reshape((-1, num_periods, num_features))
        # y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
        # X = np.tile(X, (10, 1, 1))
        # y = np.tile(y, (10, 1))
        # losses = train(X, y, args)
        # if args.show_plot:
        #     plt.plot(range(len(losses)), losses, "k-")
        #     plt.xlabel("Period")
        #     plt.ylabel("Loss")
        #     plt.show()
        