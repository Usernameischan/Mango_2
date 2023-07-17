from typing import Callable, Union, Optional, List, Tuple, Dict

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)

from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import numpy as np
import model
from sklearn.metrics import f1_score

from torch import optim
from torch.nn import BCELoss
import torch

from pickle import dump,loads


torch.manual_seed(0)
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

#1e-3 works best so far

class SplitVFL(Strategy):
    def __init__(self, num_clients, batch_size, dim_input, num_hidden_layers, train_labels, test_labels, scheduler_specs, lr=1e-3):
        """
        Strategy that implements vertical learning via a SplitNN.
        
        Args:
            num_clients: int
                Number of clients used for training.
            batch_size: int
                Batch size used in training and testing. 
            dim_inpit: int
                Dimension of input required for global model.
            train_labels: DataLoader
                DataLoader object used for storing training labels.
            test_labels: DataLoader
                DataLoader object used for storing testing labels.
            lr: float
                Learning rate for global model. 
        """
        super(SplitVFL, self).__init__()

        # store basic hyperparameters
        self.batch_size = batch_size
        self.num_clients = num_clients
        self.dim_input = dim_input
        self.num_hidden_layers = num_hidden_layers
        self.lr = lr
        self.criterion = BCELoss()
        
        # number of batches in test/train sets
        self.num_train_batches = len(train_labels)
        self.num_test_batches = len(test_labels)
        
        # store labels
        self.train_label_loader = train_labels
        self.train_label_iter = iter(self.train_label_loader)

        self.test_label_loader = test_labels
        self.test_label_iter = iter(self.test_label_loader)

        # lists used to track accuracies and other performance metrics
        self.train_acc = []
        self.test_acc = []

        self.train_correct = 0
        self.test_correct = 0

        self.test_f1 = []
        self.test_f1_accum = 0
        self.train_f1 = []
        self.train_f1_accum = 0

        self.scheduler_specs = scheduler_specs

        self.model = None
        self.client_tensors = None
        self.clients_map = None
        self.optim = None
        
        self.scheduler = None

        # Maps client id (cid) to its order in global model input
        self.order_clients = {}

        self.round = 1


        
        
    def __repr__(self) -> str:
        return "SplitVFL"

    def get_labels(self, test=False):
        """
        Get labels of train or test set.
        Args:
            test: bool
                Determines if test or train labels are fetched.
        """
        X = None
        if test: # get test labels
            try:
                X = next(self.test_label_iter)
            except StopIteration: # if completed entire test set --> eval model metrics
                metric = self.scheduler_specs['metric']
                self.test_label_iter = iter(self.test_label_loader)
                X = next(self.test_label_iter)
                self.test_acc.append(self.test_correct / len(self.test_label_loader.dataset.X))
                self.test_f1.append(self.test_f1_accum / self.num_test_batches)
                if self.scheduler_specs['on'] == "test":
                    if metric == "f1":
                        self.scheduler.step(self.test_f1[-1])
                    elif metric == "accuracy":
                        self.scheduler.step(self.test_acc[-1])
                self.test_correct = 0
                self.test_f1_accum = 0
        else:
            try:
                X = next(self.train_label_iter)
            except StopIteration: # if completed entire train set --> eval model metrics
                metric = self.scheduler_specs['metric']
                self.train_label_iter = iter(self.train_label_loader)
                X = next(self.train_label_iter)
                self.train_acc.append(self.train_correct / len(self.train_label_loader.dataset.X))
                self.train_f1.append(self.train_f1_accum / self.num_train_batches)
                if self.scheduler_specs['on'] == "train":
                    if metric == "f1":
                        self.scheduler.step(self.train_f1[-1])
                    elif metric == "accuracy":
                        self.scheduler.step(self.train_f1[-1])
                self.scheduler.step(self.train_f1[-1])
                self.train_correct = 0
                self.train_f1_accum = 0
        return X
    
    def initialize_parameters(
        self, client_manager: ClientManager
        ) -> Optional[Parameters]:
        # Initialize global model parameters
        hidden_layer_dim = (self.dim_input + 1) // 2
        layer_sizes = [self.dim_input] + [hidden_layer_dim for i in range(self.num_hidden_layers)] + [1]
        global_model = model.Net(sizes=layer_sizes)
        ndarrays = get_parameters(net=global_model)

        self.model = global_model
        self.optim = optim.Adam(self.model.parameters(),lr=self.lr)
        self.criterion = BCELoss()

        scheduler_type = self.scheduler_specs['type']
        if scheduler_type == "OnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim,mode=self.scheduler_specs['mode'])
        # return global model to clients ***not needed in VFL***
        return ndarrays_to_parameters(ndarrays)
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy,FitIns]]:
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )
        """
        Configure clients for training. 
        """
        
        rval = []

        for i, client in enumerate(clients):
            if client.cid not in self.order_clients:
                self.order_clients[client.cid] = i
            config = { 'server_round': server_round, 'order': i}
            fit_ins = FitIns(parameters, config)
            rval.append((client, fit_ins))
        # return list of tuples of client and fit instructions
        # fit instructions are just global model parameters and config.
        return rval

    # below works for clients sending one batch at a time
    def __convert_results_to_tensor(self,
        results: List[Tuple[ClientProxy, FitRes]],
        test=False
        ):
        """
        Convert intermediate results (embeddings) of client data to input tensor
        for global model
        """
        numpy_input = np.empty((self.num_clients, self.batch_size, self.dim_input // self.num_clients))

        client_tensors = [None for i in range(self.num_clients)]
        clients_map = {}

        # For each clients (i) response
        empty_samples = None
        for i, (client, fit_response) in enumerate(results):
            # get client embeddings as numpy arrays
            client_embeddings = None
            if not test:
                client_embeddings = parameters_to_ndarrays(fit_response.parameters)
            else:
                client_embeddings = loads(fit_response.metrics['test_embeddings'])

            # if the current batch is the last batch (i.e., dataset length not divisible by batch size)
            if (len(client_embeddings) != self.batch_size) and empty_samples is None:
                empty_samples = self.batch_size - len(client_embeddings)
            # for each sample's embedding
            for j, embeds in enumerate(client_embeddings):
                numpy_input[i,j,:] = embeds.astype(np.float32)
            
            # map client id to current index
            clients_map[client.cid] = i
            # add each client's batch of embeddings (size = batch_size x num features) to list
            
            ni = numpy_input[i] if empty_samples is None else numpy_input[i,:-empty_samples]
            
            # client_tensors.append(
            #     torch.tensor(ni,dtype=torch.float32,requires_grad=True if not test else False)
            # )
            client_tensors[self.order_clients[client.cid]] = torch.tensor(ni,dtype=torch.float32,requires_grad=True if not test else False)

        # client_tensors = [
        #     torch.tensor(
        #         numpy_input[(cid)],dtype=torch.float32,requires_grad=True if not test else False
        #     ) for cid in sorted(self.order_clients.keys()) # would need to change this if clients send embeddings at different times or if failures
        # ]
        # create global model input --> size = batch_size x self.dim_input

        gbl_model_input = torch.cat(client_tensors, 1)

        if not test:
            self.client_tensors = client_tensors
            self.clients_map = clients_map

        return gbl_model_input

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Completes forward pass of global model for training and update global model params.
        
        Args:
            server_round: int
                Current server round.
            results: List of results
                Results returned by clients -- embeddings.
            failures: List of failures provided by clients
                -
        Returns:
            Tuple
                Returns global model parameters and loss metric score.
        """
        # convert all intermediate results from clients as 
        # input to the global model
        gbl_model_input = self.__convert_results_to_tensor(results=results)
        gbl_model_output = self.model(gbl_model_input)

        # get labels
        labels = self.get_labels(test=False).float()
        # zero gradients
        self.optim.zero_grad()
        # compute loss
        loss = self.criterion(
            gbl_model_output,
            labels
        )
        # backpropagation and update
        loss.backward()
        self.optim.step()
        
        # get predictions
        preds = torch.round(gbl_model_output)
        # number of correct preds in batch
        num_correct = (
                preds == labels
            ).float().sum().squeeze().item()

        self.train_correct += num_correct
        f1 = f1_score(labels.numpy(), preds.detach().numpy())
        self.train_f1_accum += f1
        
        return (ndarrays_to_parameters(
            get_parameters(self.model)
        ), {'loss': str(loss.item()), })

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure clients for testing.
        """
        # sample all clients
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.num_clients
        )
        # empty config
        config = {'server_round': server_round}
        # instructions for clients
        eval_ins = []
        
        for client in clients:
            idx = self.clients_map[client.cid]
            # provide gradient of loss fxn wrt clients inputs to model (their outputs) to each respective client.
            # tensor = self.client_tensors[idx]
            tensor = self.client_tensors[self.order_clients[client.cid]]
            
            ins = EvaluateIns(
                ndarrays_to_parameters(tensor.grad.numpy()),
                config
            )
            eval_ins.append(ins)
        return [(client, eins) for (client,eins) in zip(clients, eval_ins)]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Compute model performance metrics on test set. 
        """
        criterion = BCELoss()
        # convert client inputs to global model input
        gbl_model_input = self.__convert_results_to_tensor(results,test=True)
        
        with torch.no_grad():
            # compute model output and get metrics
            gbl_model_output = self.model(gbl_model_input).float()
            labels = self.get_labels(test=True).float()
            loss = criterion(
                gbl_model_output,
                labels
            )
            # obtain predictions
            preds = torch.round(gbl_model_output)
            # number correct preds in current batch
            num_correct = (
                preds == labels
            ).float().sum().squeeze().item()
            # compute f1
            f1 = f1_score(labels.numpy(), preds.numpy())

            self.test_f1_accum += f1
            self.test_correct += num_correct
        # return dummy evaluation dictionaru
        return (server_round, {'None': str(None)})

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters on central dataset."""
        pass
    # return model
    def get_model(self):
        return self.model.state_dict()


