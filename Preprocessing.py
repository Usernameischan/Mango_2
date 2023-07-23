import pandas as pd
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
# from icecream import ic

# encoding the timestamp data cyclically. See Medium Article.
def process_data(source):

    df = pd.read_excel(source)
    
    df.drop(df[df['Inventory quantity'] == '********  '].index, inplace=True)
    df.drop(df[df['Sale quantity'] == '********  '].index, inplace=True)
    df = df.loc[df['Item number'] == 20280029, ['Date', 'Store Number', 'Inventory quantity']].reset_index(drop=True)
        
    day = np.array(df['Date'].dt.day)
    month = np.array(df['Date'].dt.month)

    days_in_month = 30
    month_in_year = 12


    df['sin_day'] = np.sin(2*np.pi*day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*month/month_in_year)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
        
    df.plot.scatter('sin_day','cos_day', ax=axes[0]).set_aspect('equal')
    df.plot.scatter('sin_month','cos_month', ax=axes[1]).set_aspect('equal')
    
    size_3853_size = len(df[df['Store Number'] == 3853])
    size_3349_size = len(df[df['Store Number'] == 3349])
    size_607_size = len(df[df['Store Number'] == 607])

    train_3853 = df[df['Store Number'] == 3853].sort_values('Date').reset_index(drop=True)[:int(size_3853_size * 0.8)]
    train_3349 = df[df['Store Number'] == 3349].sort_values('Date').reset_index(drop=True)[:int(size_3349_size * 0.8)]
    train_607 = df[df['Store Number'] == 607].sort_values('Date').reset_index(drop=True)[:int(size_607_size * 0.8)]

    test_3853 = df[df['Store Number'] == 3853].sort_values('Date').reset_index(drop=True)[int(size_3853_size * 0.8):]
    test_3349 = df[df['Store Number'] == 3349].sort_values('Date').reset_index(drop=True)[int(size_3349_size * 0.8):]
    test_607 = df[df['Store Number'] == 607].sort_values('Date').reset_index(drop=True)[int(size_607_size * 0.8):]

    train = pd.concat([train_3853, train_3349, train_607]).reset_index(drop=True)
    test = pd.concat([test_3853, test_3349, test_607]).reset_index(drop=True)

    return train, test

train, test = process_data('Data/reach_data22.xlsx')
train.to_csv(r'Data/train_reach_dataset.csv', index=False)
test.to_csv(r'Data/test_reach_dataset.csv', index=False)
