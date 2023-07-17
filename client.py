import flwr as fl 
import torch
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from flwr.common import parameters_to_ndarrays
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pickle as pi
import json

import time


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,cid, net, trainloader,testloader, optimizer):
        """
        Class that represents a client in Flower.

        Args:
            cid: int
                Integer representing ClientID
            net: PyTorch NN
                A PyTorch NN to be used for training with this client.
            trainloader: DataLoader
                DataLoader object to use while training model.
            testloader: DataLoader
                DataLoader object to use while training model.
            optimizer: PyTorch Optimizer
                Optimizer to use while updating model parameters. (e.g. Adam)
        """
        super(FlowerClient,self).__init__()
        # client id
        self.cid = cid
        # client network
        self.net = net
        # train loader and iterator
        self.trainloader = trainloader
        self.trainiter = iter(trainloader)
        # test loader and iterator
        self.test_loader = testloader 
        self.testiter = iter(self.test_loader)

        # optimizer (
        self.optimizer = optimizer
        # outputs from previous round
        self.outputs = None

        self.myorder = None
        
    # Gets model parameters
    def get_parameters(self, config):
        return [
            val.cpu.numpy() for _,val in self.net.state_dict().items()
        ]


    def fit(self,parameters,config):
        """
        Completes forward pass of one batch.

        Args:
            parameters: Parameters
                Parameters of global model. ***Not needed in VFL***
            config: dict
                Configuration dictionary provided by server that contains
                configuration information like current round.

        Returns:
            Tuple
                Returns embeddings of train batch to server and dummy metrics
        """
        if self.myorder is None and 'order' in config:
            self.myorder = config['order']
        # Read values from config
        server_round = config['server_round']
        # get batch
        try:
            X = next(self.trainiter)
        except StopIteration:
            self.trainiter = iter(self.trainloader)
            X = next(self.trainiter)
        outputs = self.net(X.float())
        self.outputs = outputs
        
        return [x for x in outputs.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        """
        Update model parameters and complete forward pass of one batch
        of test data.

        Args:
            parameters: Parameters
                Parameters of global model. ***Not needed in VFL***
            config: dict
                Configuration dictionary provided by server that contains
                configuration information like current round
        Returns:
            Tuple
                Returns embeddings of test batch in metrics dictionary.
        """
        self.outputs.backward(torch.tensor(np.array(parameters)))
        self.optimizer.step()
        try:
            X = next(self.testiter)
        except StopIteration:
            self.testiter = iter(self.test_loader)
            X = next(self.testiter)
        with torch.no_grad():
            outputs = self.net(X.float()).numpy()
        bytes_outputs = pi.dumps([ x for x in outputs])
        return 0., 0, {'test_embeddings': bytes_outputs}
    
    # get model
    def get_model(self):
        return self.net.state_dict()

    def get_order(self):
        return self.myorder
    
if __name__ == "__main__":
    import argparse
    import pickle
    from utils import ShoppersDataset, ClientIdentifier
    import model
    from torch.optim import Adam

    # set seed for consitency
    torch.manual_seed(0)
    
    # accept client id arguement from cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument("cid",type=int)
    parser.add_argument("-d","--datafile",type=str)
    args = parser.parse_args()

    infile = './change_data_tmp.pt'
    cid    = args.cid

    # cid maps to one of 'brand','company','category'
    ci = ClientIdentifier()
    client_type = ci.get_client_from_cid(cid=cid)

    # obtain training data from file saved by server
    with open(infile,'rb') as f:
        data = pickle.load(f)
    
    # Get train and test dataloader objects
    train_dataloader = data['data'][client_type]['train']
    test_dataloader  = data['data'][client_type]['test']

    # load client's model defintion
    client_dict = None
    with open("model_definitions.json",'r') as f:
        client_dict = json.load(f)[client_type]
    

    # learning rate
    lr     = client_dict['lr']
    output_dim = client_dict['output_dim']
    num_hidden_layers = client_dict['hidden']
    input_dim = client_dict['input_dim']

    
    # # fix model with size 6 output dimensions
    hidden_layer_dim = (input_dim + output_dim) // 2
    layers = [input_dim] + [hidden_layer_dim for i in range(num_hidden_layers)] + [output_dim]
    model = model.Net(layers)

    Client = FlowerClient(
        cid=str(args.cid), 
        net = model, 
        trainloader=train_dataloader, 
        testloader=test_dataloader, 
        optimizer = Adam(
            params=model.parameters(), 
            lr=lr
        )
    )

    # # start client and connect to server
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=Client
    )

    # Get model's state dict and save for testing purposes after training. 
    client_model = Client.get_model()
    torch.save(client_model, f'./models/model_{args.cid}.pt')

    info = {
        'cid': args.cid,
        'order': Client.get_order(),
        'hps': {
            'output_dim': output_dim,
            'lr': lr,
            'output_activation': 'sig',
            'optim_type': 'adam'
        }
    }
    with open(f"./models/model_{args.cid}_info.pt",'wb') as f:
        pi.dump(info, f)
    

