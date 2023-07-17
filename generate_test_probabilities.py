import pandas as pd 
from utils import ShoppersDataset, ClientIdentifier
from torch.utils.data import DataLoader
from model import Net
import torch
from pickle import load
import json
import argparse

def get_trained_model(client_type, ci, global_dim=None):
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

def get_client_order(cids):
    rval = {}
    for cid in cids:
        with open(f"./models/model_{cid}_info.pt","rb") as f:
            rval[cid] = load(f)['order']
    return rval

def get_client_hps(cid):
    with open(f'./models/model_{cid}_info.pt','rb') as f:
        d = load(f)
        return d['hps']

if __name__ == "__main__":
    
    mdl_definitions = None
    with open('model_definitions.json','r') as f:
        mdl_definitions = json.load(fp=f)

    num_rounds = mdl_definitions['global']['num_rounds']
    bs         = mdl_definitions['global']['batch_size']
    scheduler_specs = mdl_definitions['global']['scheduler']
    # get test data
    test_data = pd.read_csv("./data/tmp_test_data.csv", index_col=0)
    test_data = test_data.drop(['repeater', 'date', 'productmeasure'], axis=1)

    comp_cols, brand_cols, cat_cols = [],[],[]
    for num, col in enumerate(test_data.columns):
        if 'brand' in col: 
            brand_cols.append(col)
        elif 'cat' in col:
            cat_cols.append(col)
        elif 'comp' in col:
            comp_cols.append(col)
        else:
            comp_cols.append(col)
    
    company_client = test_data[comp_cols].to_numpy()
    brand_client = test_data[brand_cols].to_numpy()
    category_client = test_data[cat_cols].to_numpy()

    company_dl = DataLoader(
        dataset = ShoppersDataset(company_client),
        batch_size=company_client.shape[0],
        shuffle=False
    )
    brand_dl = DataLoader(
        dataset = ShoppersDataset(brand_client),
        batch_size=brand_client.shape[0],
        shuffle=False
    )
    category_dl = DataLoader(
        dataset = ShoppersDataset(category_client),
        batch_size=category_client.shape[0],
        shuffle=False
    )

    ci = ClientIdentifier()
    company_cid = ci.get_cid_from_client(client_type='company')
    brand_cid = ci.get_cid_from_client(client_type='brand')
    category_cid = ci.get_cid_from_client(client_type='category')

    dim = 0
    for client in mdl_definitions:
            if client != 'global':
                dim += mdl_definitions[client]['output_dim']

    gblr = mdl_definitions['global']['lr']

    company_model_hps  = mdl_definitions['company']
    brand_model_hps    = mdl_definitions['brand']
    category_model_hps = mdl_definitions['category']

    
    company_model = get_trained_model(client_type='company', ci=ci)
    brand_model =  get_trained_model(client_type='brand', ci=ci)
    category_model = get_trained_model(client_type='category', ci=ci)

    global_model = get_trained_model(client_type='global', ci=ci, global_dim=dim)

    client_order = get_client_order([company_cid, brand_cid, category_cid])

    with torch.no_grad():
        company_outputs = company_model(next(iter(company_dl)).float())
        brand_outputs = brand_model(next(iter(brand_dl)).float())
        category_outputs = category_model(next(iter(category_dl)).float())

        gbl_model_inputs = [None for i in range(3)]

        gbl_model_inputs[client_order[company_cid]] = company_outputs
        gbl_model_inputs[client_order[brand_cid]] = brand_outputs
        gbl_model_inputs[client_order[category_cid]] = category_outputs

        gbl_model_inputs = torch.cat(gbl_model_inputs,dim=1)

        gbl_model_outputs = global_model(gbl_model_inputs)

        preds = gbl_model_outputs.numpy()
    
    print(len(preds))
    ids = pd.read_csv("./data/testHistory.csv").id 
    df = pd.DataFrame(data={"id": ids, "repeatProbability": preds.squeeze()})
    
    keys = sorted(company_model_hps.keys())
    
    names = []
    for chps in [company_model_hps,brand_model_hps,category_model_hps]:
        client_model_name = ''
        for i,key in enumerate(keys):
            delim = '_' if i < (len(keys)-1) else ''
            client_model_name += f'{chps[key]}{delim}'
        names.append(client_model_name)
    
    scheduler_name = "_".join(scheduler_specs[k] for k in sorted(scheduler_specs))
    names.append(f"{mdl_definitions['global']['hidden']}_{gblr}_{bs}_{num_rounds}_{scheduler_name}")

    fed_mdl_name = '__'.join(names)

    fname = f'./eval/probs/{fed_mdl_name}.csv'
    df.to_csv(fname,index=False)

    from csv import writer
    with open('eval/evaluations_scores.csv','a') as fobj:
        writer_object = writer(fobj)
        writer_object.writerow([fname,fed_mdl_name])



    
    