from model import Net
import torch
from utils import ShoppersDataset,ClientIdentifier
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
from pickle import load
import json

# get a given client's test data
def load_client_data(data_dict, cid, ci):
    client_type = ci.get_client_from_cid(cid=cid)
    return data_dict['data'][client_type]['test']

def get_trained_model(client_type,ci,global_dim=None):
    # load model definitions file 
    mdl_def = None
    with open('./model_definitions.json','r') as f:
        mdl_def = json.load(fp=f)[client_type]

    input_dim = global_dim if (global_dim != None and client_type == 'global') else mdl_def['input_dim']

    # get hidden layer dimensions (assuming same height)
    hidden_layer_dim = (input_dim + mdl_def['output_dim']) // 2
    # create layer list and load model.
    layers = [input_dim] + [hidden_layer_dim for i in range(mdl_def['hidden'])] + [mdl_def['output_dim']]
    model = Net(layers)
    name = "global_model" if client_type == 'global' else f"model_{ci.get_cid_from_client(client_type)}"
    model.load_state_dict(torch.load(f=f'./models/{name}.pt'))
    # set model for evaluation stage
    model.eval()
    return model


# get all client datasets, models, and targets
def get_client_info(cids, infile, ci):

    # load data saved earlier (in Server.py)
    with open(infile,'rb') as f:
        data = load(f)

    # get labels
    labels = data['test_labels']

    # get client_data, clients loaded in order of cids given
    client_data = [
        load_client_data(data, cid, ci=ci) for cid in cids
    ]
    # get client models
    client_models = [
        get_trained_model(
            client_type=ci.get_client_from_cid(cid=cid),
            ci=ci
        )
        for cid in cids
    ]
    return client_data, client_models, labels

def get_client_order(cids):
    rval = {}
    for cid in cids:
        with open(f"./models/model_{cid}_info.pt","rb") as f:
            d = load(f)
            rval[cid] = d['order']

    return rval


def main(num_clients, infile):

    dim = 0
    mdl_dfns = None
    with open('model_definitions.json','r') as f:
        mdl_dfns = json.load(f)
        for client in mdl_dfns:
            if client != 'global':
                dim += mdl_dfns[client]['output_dim']

    
    ci = ClientIdentifier()
    global_model = get_trained_model('global', ci,global_dim=dim)

    client_data, client_models, labels_dl = get_client_info(cids=[0, 1, 2], infile=infile, ci=ci)

    labels = iter(labels_dl)

    client_iters = [iter(dl) for dl in client_data]

    criterion = torch.nn.BCELoss()
    
    pred_scores = []
    
    client_order = get_client_order(cids=[0,1,2])

    with torch.no_grad():
        while True:
            try:
                # get client inputs
                Client0Inputs = next(client_iters[0]).float()
                Client1Inputs = next(client_iters[1]).float()
                Client2Inputs = next(client_iters[2]).float()
                # get labels
                ls = next(labels)
                # get embeddings
                Client0Embeddings = client_models[0](Client0Inputs).detach()
                Client1Embeddings = client_models[1](Client1Inputs).detach()
                Client2Embeddings = client_models[2](Client2Inputs).detach()

                input_tensor = [None for i in range(3)]
                input_tensor[client_order[0]] = Client0Embeddings
                input_tensor[client_order[1]] = Client1Embeddings
                input_tensor[client_order[2]] = Client2Embeddings
                # create input to global model
                gbl_mdl_inputs = torch.cat(input_tensor, dim = 1 )
                # get outputs
                gbl_mdl_outputs = global_model(gbl_mdl_inputs)
                
                pred_scores += gbl_mdl_outputs.squeeze().tolist()
            except StopIteration:
                # print("Successfully iterated through test set", len(pred_scores) == labels_dl.dataset.X.shape[0])
                break
    # Get AUC of ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels_dl.dataset.X.squeeze(), y_score = pred_scores)
    roc_auc = metrics.auc(fpr,tpr)
    display = metrics.RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc=roc_auc)
    display.plot()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numclients",type=int)
    parser.add_argument("-d", "--datafile",type=str)

    args = parser.parse_args()

    NUM_CLIENTS = args.numclients
    infile = args.datafile
    
    main(
        num_clients=3,
        infile="./change_data_tmp.pt"
    )
    pass



