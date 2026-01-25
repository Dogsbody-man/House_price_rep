coloms_to_drop = ['Utilities', 'WoodDeckSF', '2ndFlrSF', 'OpenPorchSF', 'HalfBath', 'LotArea',
       'BsmtFullBath', 'BsmtUnfSF', 'BedroomAbvGr', 'ScreenPorch', 'PoolArea',
       'MoSold', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MiscVal', 'Id',
       'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch',
       'KitchenAbvGr', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt', ]

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch
import sys
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
original_train = train_df.copy()
original_test = test_df.copy()
target = train_df['SalePrice']

def processing_passes(train, test, )

