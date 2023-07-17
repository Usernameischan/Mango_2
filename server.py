import Strategy as stgy
import pickle
from utils import ShoppersDataset,load_datasets,save_data
import time
from torch import save

from torch.optim import Adam
from torch.utils.data import DataLoader
import flwr as fl

import argparse
import json

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument("-bs","--batchsize",type=int)
    # parser.add_argument("-n", "--numclients",type=int)
    # parser.add_argument("-do","--dataoutput",type=str)
    # parser.add_argument("-nr","--numrounds",type=int)
    # parser.add_argument("-f","--trainfile",type=str)
    
    # args = parser.parse_args()
    NUM_CLIENTS = 3
    outfile = './change_data_tmp.pt'
    infile = './data/tmp_train_data.csv'
    
    with open('model_definitions.json','r') as f:
        model_definitions = json.load(f)
    
    # number of batches to iterate through
    NUM_ROUNDS = model_definitions['global']['num_rounds']
    BATCH_SIZE = model_definitions['global']['batch_size']

    # get data and save
    DATA = save_data(data_path=infile, batch_size=BATCH_SIZE, outfile=outfile)


    lr = model_definitions['global']['lr']
    input_dim = 0
    for mdl in model_definitions:
        if mdl != 'global':
            input_dim += model_definitions[mdl]['output_dim']

    scheduler_specs = model_definitions['global']['scheduler']
    
    # create strategy
    CustomStrategy = stgy.SplitVFL(
        num_clients = NUM_CLIENTS, 
        batch_size = BATCH_SIZE, 
        dim_input = input_dim, # 6 outputs for three clients
        num_hidden_layers = model_definitions['global']['hidden'],
        train_labels = DATA['train_labels'],
        test_labels = DATA['test_labels'],
        scheduler_specs = scheduler_specs
    )
    # start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),
        strategy = CustomStrategy
    )

    # get and save global model for testing purposes.
    global_model = CustomStrategy.get_model()
    save(global_model, "./models/global_model.pt")
    
    # import matplotlib.pyplot as plt 

    # fig, axs = plt.subplots(1, 2, sharey=True)
    
    # axs[0].plot(CustomStrategy.train_acc, label='Train Accuracy')
    # axs[1].plot(CustomStrategy.test_acc, label='Test Accuracy')
    # axs[1].plot(CustomStrategy.test_f1, label='Test F1-Score')
    # axs[0].legend()
    # axs[1].legend()
    # plt.show()
