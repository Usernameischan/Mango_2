import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch.optim as optim
import os, sys, math, copy
os.chdir(r'C:\Users\uoon9\GitHub\Mango_2\data\03. Dataset_CNC\dataset\CNC 학습통합데이터_1209')



warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, input_dim=48, hidden_dim=20, output_dim=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def train(model, X_train_tensor, Y_train_tensor, num_epochs=1000):
    """Train the model on the training set."""
    # 손실함수 정의
    criterion = nn.BCELoss()

    # 옵티마이저 정의
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Forward
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 매 100번째 에포취마다 로그 출력
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

def test(model, X_test_tensor, Y_test_tensor):
    """Validate the model on the test set."""
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == Y_test_tensor).float().mean()
        loss = nn.BCELoss()(outputs, Y_test_tensor)
    return loss, accuracy

# Function Load CNC DATA

def load_data():
    # 데이터 불러오기
    X_train = pd.read_csv('X_train.csv', header = None, encoding = 'utf-8')
    X_test = pd.read_csv('X_test.csv', header = None, encoding = 'utf-8')
    Y_train = pd.read_csv('Y_train.csv', header = None, encoding = 'utf-8')
    Y_test = pd.read_csv('Y_test.csv', header = None, encoding = 'utf-8')

    # 데이터 변환 함수
    def data_transform(df):
        return np.array(df)

    # numpy 배열로 변환
    X_train_np = data_transform(X_train)
    X_test_np = data_transform(X_test)
    Y_train_np = data_transform(Y_train)
    Y_test_np = data_transform(Y_test)

    # PyTorch 텐서로 변환
    X_train_tensor = torch.Tensor(X_train_np)
    X_test_tensor = torch.Tensor(X_test_np)
    Y_train_tensor = torch.Tensor(Y_train_np)
    Y_test_tensor = torch.Tensor(Y_test_np)
    
    return X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor


net = Net().to(DEVICE)
X_train_tensor, X_test_tensor, Y_train_tensor, Y_test_tensor = load_data()


'''
2. Flower 라이브러리를 사용하여 Federated learing 작동하도록 구성
2-1. pytorch모델을 flower 모델로 변환하기 위해 Numpyclient 하위클래스를 상속받는, FlowerClient 클래스 정의
2-2. FlowerClient 클래스는 get_parameters 메서드와 set_parameters 메서드를 사용하여, pytorch 모델의 가중치를 numpy 배열로 가져오고 설정
2-3. get_parameters 메서드 : 파이토치로 생성한 모델의 가중치를 numpy 배열로 반환
    set_parameters 메서드 : get_parameters에서 반환된 numpy 배열로부터 모델 가중치를 로드.
2-4. fit 메서드 : 모델 가중치를 사용하여 클라이언트에서 모델을 학습, 완료되면 학습된 가중치를 반환
2-5. evaluate 메서드 : 모델 가중치를 사용하여 클라이언트에서 모델을 평가, 평가 결과를 반환

'''
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config, **kwargs):
        self.set_parameters(parameters)
        train(net, X_train_tensor, Y_train_tensor, num_epochs=1)
        return self.get_parameters(None), len(X_train_tensor), {}

    def evaluate(self, parameters, config, **kwargs):
        self.set_parameters(parameters)
        loss, accuracy = test(net, X_test_tensor, Y_test_tensor)
        return loss.item(), len(X_test_tensor), {"accuracy": accuracy.item()}


'''
3. 위에서 구현한 FlowerClient 클래스를 사용하여 시작
'''
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
