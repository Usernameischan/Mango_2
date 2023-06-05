from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,random_split
import torch
import pandas as pd
import numpy as np 
from pickle import dump
import json
from io import BytesIO
import os 
import zipfile
from zipfile import ZipFile
import requests

class ShoppersDataset(Dataset):
    """
        The class used to represent a given client's data set in VFL (no labels).

        Attributes
        __________
        X: numpy.ndarray
            A numppy array representing client's dataset. Each row represents a consumer's features.
        """
    def __init__(self,X):
        self.X = X
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index,:]

def create_train_test(df, train_index, test_index, batch_size, cols=None):
    """
    Create train and test DataLoader objects using specified parameters.

    Args:
        df: pandas.DataFrame
            DataFrame object containing data to be used for training and testing purposes.
        cols: list
            A subset of columns of `df` to include when creating train/test DataLoaders.
        train_index: array-like
            An iterable that contains indices representing training samples in `df`. 
        test_index: array-like
            An iterable that contains indices representing testing samples in `df`. 
        batch_size: int
            Batch size to use while training/testing models.
    
    Returns:
        tuple(DataLoader)
            A tuple that includes the train and test DataLoaders. 
    """
    torch.manual_seed(seed=0)
    
    if cols != None:
        train_ds = ShoppersDataset(
            df.loc[train_index][cols].copy().to_numpy()
        )
        test_ds = ShoppersDataset(
            df.loc[test_index][cols].copy().to_numpy()
        )
        train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
        return train_dl,test_dl
    else:
        train_ds = ShoppersDataset(
            df.loc[train_index].copy().to_numpy()
        )
        test_ds = ShoppersDataset(
            df.loc[test_index].copy().to_numpy()
        )
        train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
        return train_dl,test_dl

def load_datasets(data_path,batch_size):
    """
    Creates vertically-split datasets to use for vertical federated learning.

    Args:
        data_path: str
            Path to the .csv file that includes data for training and testing.
        batch_size: int
            Batch size to use while training/testing models.
    
    Returns:
        dict
            A dictionary that contains the vertically-split datasets for use in federated learning along 
            with associated train/test labels. 
    """
    # df = pd.read_csv(data_path) ############################# 바꾸기 data_path = zip file0
    data_filename = []
    data=[] 
    with ZipFile(data_path, 'r') as zipObj:
        listOfFileNames = zipObj.namelist()
        for fileName in listOfFileNames:
            if fileName.endswith('csv'): 
                # print(fileName)
                zipRead = zipObj.read(fileName)
                curr_df = pd.read_csv(BytesIO(zipRead))
                data.append(curr_df)
                data_filename.append(fileName)

    # for filename, d in zip(data_filename, data): 
    #     print(filename)
    #     display(d.head(5))
    
    # Label Encoding
    target_code = {'RRR':0,'GGG':1,'BBB':2}
    data[0]['target'] = data[0]['target'].map(target_code)
    
    # 하위 절반 행 삭제(메모리 이슈로 인해, 모델 정상작동 확인 후 제거할 것.)
    data[0] = data[0].iloc[len(data[0])//2:, :]
    data[1] = data[1].iloc[len(data[1])//2:, :]
    data[2] = data[2].iloc[len(data[2])//2:, :]
    
    data[0] = data[0].iloc[:, len(data[0].columns)//2:]
    data[1] = data[1].iloc[:, len(data[1].columns)//2:]
    data[2] = data[2].iloc[:, len(data[2].columns)//2:]
    
    train_index, test_index = train_test_split(data[0].index.values,test_size=0.2,stratify=data[0]['target'])
    train_labels, test_labels = create_train_test(
        df=data[0], train_index=train_index, test_index=test_index, batch_size=batch_size, cols=['target']
    )
    data[0].drop('target', axis=1, inplace=True)
    data[1].drop('target', axis=1, inplace=True)
    data[2].drop('target', axis=1, inplace=True)





    train_comp_dl, test_comp_dl = create_train_test(data[0].iloc[:,1:], train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_cat_dl, test_cat_dl = create_train_test(data[1].iloc[:,1:], train_index=train_index, test_index=test_index, batch_size=batch_size)
    train_brand_dl, test_brand_dl = create_train_test(data[2].iloc[:,1:], train_index=train_index, test_index=test_index, batch_size=batch_size)


    # # Get train/test indices
    # train_index, test_index = train_test_split(df.index.values,test_size=0.2,)
    # # treating labels as dataloaders --> simplifies VFL training.
    # # get labels
    # train_labels, test_labels = create_train_test(
    #     df=df, cols=['target'], train_index=train_index, test_index=test_index, batch_size=batch_size
    # )
    # # drop repeater column (clients don't have labels)
    # df = df.drop('target',axis=1)

    # # Split's data into three types of clients: brand-,category-, and company-based.
    # comp_cols, brand_cols, cat_cols = [],[],[]
    # for i,col in enumerate(df.columns):
    #     if 'brand' in col: 
    #         brand_cols.append(col)
    #     elif 'cat' in col:
    #         cat_cols.append(col)
    #     elif 'comp' in col:
    #         comp_cols.append(col)
    #     else:
    #         comp_cols.append(col)

    # create dataloaders for each client
    # train_comp_dl, test_comp_dl = create_train_test(df, cols=comp_cols,  train_index=train_index, test_index=test_index, batch_size=batch_size)
    # train_cat_dl, test_cat_dl = create_train_test(df, cols=cat_cols,   train_index=train_index, test_index=test_index, batch_size=batch_size)
    # train_brand_dl, test_brand_dl = create_train_test(df, cols=brand_cols, train_index=train_index, test_index=test_index, batch_size=batch_size)

    train_comp_shape = train_comp_dl.dataset.X.shape
    test_comp_shape = test_comp_dl.dataset.X.shape

    train_brand_shape = train_brand_dl.dataset.X.shape
    test_brand_shape = test_brand_dl.dataset.X.shape 

    train_cat_shape = train_cat_dl.dataset.X.shape
    test_cat_shape = test_cat_dl.dataset.X.shape


    assert (
        train_comp_shape[0] == train_cat_shape[0] ==  train_brand_shape[0] == train_labels.dataset.X.shape[0] and 
        test_comp_shape[0] == test_cat_shape[0] ==  test_brand_shape[0] == test_labels.dataset.X.shape[0]
    )

    assert (
        train_comp_shape[1] == test_comp_shape[1] and 
        train_brand_shape[1] == test_brand_shape[1] and
        train_cat_shape[1] == test_cat_shape[1]
    )

    mdl_dfns = None
    with open('model_definitions.json','r') as f:
        mdl_dfns = json.load(f)
    
    mdl_dfns['company']['input_dim'] = train_comp_shape[1]
    mdl_dfns['category']['input_dim'] = train_cat_shape[1]
    mdl_dfns['brand']['input_dim'] = train_brand_shape[1]

    with open('model_definitions.json','w') as f:
        json.dump(obj=mdl_dfns, fp=f,indent=4)

    rval = {
        'data': {
            'company': {'train': train_comp_dl, 'test': test_comp_dl},
            'brand': {'train': train_brand_dl, 'test': test_brand_dl},
            'category': {'train': train_cat_dl, 'test': test_cat_dl}
        },
        'train_labels': train_labels, 'test_labels': test_labels, 
        'batch_size': batch_size
    }
    return rval 

def save_data(data_path,batch_size,outfile):
    """
    Creates client datasets and saves to pickled file for future use.

    Args:
        data_path: str
            Path to the .csv file that includes data for training and testing.
        batch_size: int
            Batch size to use while training/testing models.
        outfile: str
            Output file destination of pickled dataset dict. 
    
    Returns:
        dict
            A dictionary that contains the vertically-split datasets for use in federated learning along 
            with associated train/test labels. 
    """
    data = load_datasets(
        data_path=data_path, 
        batch_size=batch_size
    )
    with open(outfile,'wb') as f:
        dump(data, f)
    
    return data

class ClientIdentifier:

    def __init__(self):
        self.__cid_to_client_map = {
            0: 'company',
            1: 'category',
            2: 'brand'
        }
        self.__client_to_cid = {
            'company': 0,
            'category': 1,
            'brand': 2
        }
    def get_cid_from_client(self,client_type):
        return self.__client_to_cid[client_type]
    
    def get_client_from_cid(self,cid):
        return self.__cid_to_client_map[cid]

