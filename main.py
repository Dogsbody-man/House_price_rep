import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from xgboost import XGBRegressor
from omegaconf import OmegaConf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from config import config


## -- Dataset --

class Dataset:

    def __init__(self, config):
        self.config = config
        self.target = self.config.data.target_column
        self.scaler = None
        self.encoder = None
        # Features
        self.num_features = None
        self.cat_features = None
        self.asymmetrical_col = None
        # Statistics
        self.neighborhood_means = None
        self.bathfull_median = None
        self.bathhalf_median = None
        self.MasVnrArea_mode = None
        self.MSZoning_mode = None
        self.Utilities_mode = None
        self.Functional_mode = None
        self.Exterior1st_mode = None
        self.Exterior2nd_mode = None
        self.SaleType_mode = None
    
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # -- Outlier Removal --
        df = df.drop(df[(df['GrLivArea']>4000) 
                        & (df['SalePrice']<300000)].index).reset_index(drop=True)
        df = df.drop(df[(df['GarageArea'] > 1000) 
                        & (df['SalePrice'] < 300000)].index).reset_index(drop=True)

        # -- Statistics --
        self.neighborhood_means = df.groupby('Neighborhood')['LotFrontage'].mean()
        self.bathfull_median = df['BsmtFullBath'].median()
        self.bathhalf_median = df['BsmtHalfBath'].median()
        self.MasVnrArea_mode = df['MasVnrArea'].mode()[0]
        self.MSZoning_mode = df['MSZoning'].mode()[0]
        self.Utilities_mode = df['Utilities'].mode()[0]
        self.Functional_mode = df['Functional'].mode()[0]
        self.Exterior1st_mode = df['Exterior1st'].mode()[0]
        self.Exterior2nd_mode = df['Exterior2nd'].mode()[0]
        self.SaleType_mode = df['SaleType'].mode()[0]

        df['MSSubClass'] = df['MSSubClass'].apply(str)
        df['YrSold'] = df['YrSold'].apply(str)
        df['MoSold'] = df['MoSold'].apply(str)
        
        # -- Drop columns --
        df.drop(['Id'], axis=1, inplace=True)

        # -- Normilization by skewness and kurtosis--
        self.num_features = [col for col in df.columns if df[col].dtype != object and col != self.target]
        self.cat_features = [col for col in df.columns if df[col].dtype == object]

        # -- Skewed features --
        self.asymmetrical_col = []
        for col in self.num_features:
            if abs(df[col].skew()) > 1 or abs(df[col].kurt()) > 3:
                self.asymmetrical_col.append(col)

        for c in ['SalePrice', 'GrLivArea', 'WoodDeckSF']:
            if c in self.asymmetrical_col:
                self.asymmetrical_col.remove(c)
        
        if self.config.general.selected_model !='catboost':
        # -- Encoder --
            self.encoder = OneHotEncoder(handle_unknown='ignore')
            self.encoder.fit(df[self.cat_features])
        # -- Scaler --
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.num_features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # -- Fill missings --
        df['MSSubClass'] = df['MSSubClass'].apply(str)
        df['YrSold'] = df['YrSold'].apply(str)
        df['MoSold'] = df['MoSold'].apply(str)

        df['PoolQC'] = df['PoolQC'].fillna('None')
        df['MiscFeature'] = df['MiscFeature'].fillna('None')
        df['Alley'] = df['Alley'].fillna('None')
        df['Fence'] = df['Fence'].fillna('None')
        df['MasVnrType'] = df['MasVnrType'].fillna('None')
        df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            df[col] = df[col].fillna('None')
        for col in ['BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtExposure', 'BsmtCond']:
            df[col] = df[col].fillna('None')
        df['MasVnrArea'] = df['MasVnrArea'].fillna(self.MasVnrArea_mode)
        df['MSZoning'] = df['MSZoning'].fillna(self.MSZoning_mode)
        df['Utilities'] = df['Utilities'].fillna(self.Utilities_mode)
        df['Functional'] = df['Functional'].fillna(self.Functional_mode)
        df['Exterior1st'] = df['Functional'].fillna(self.Exterior1st_mode)
        df['Exterior2nd'] = df['Functional'].fillna(self.Exterior2nd_mode)
        df['SaleType'] = df['SaleType'].fillna(self.SaleType_mode)
        df['KitchenQual'] = df['KitchenQual'].fillna('TA')
        df['GarageArea'] = df['GarageArea'].fillna(0)
        for col in ['BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1']:
            df[col] = df[col].fillna(0)
        df['LotFrontage'] = df['LotFrontage'].fillna(df['Neighborhood'].map(self.neighborhood_means))
        df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(self.bathhalf_median)
        df['BsmtFullBath'] = df['BsmtFullBath'].fillna(self.bathfull_median)
        df['Electrical'] = df['Electrical'].fillna('SBrkr')

        # -- Log transform --
        unique_val = df[self.asymmetrical_col].nunique()
        unique_val = unique_val[unique_val > 15].index
        for col in unique_val:
            df[col] = np.log1p(df[col]) 


        # -- Encoding -- 
        if self.encoder is not None:
            encoded = self.encoder.transform(df[self.cat_features]).toarray()
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(self.cat_features),
                index=df.index
            )
            df = pd.concat([df.drop(columns=self.cat_features), encoded_df], axis=1)

        # -- Scaling --
        if self.scaler is not None:
            scale_cols = [c for c in self.num_features if c in df.columns]
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df

## -- DNN -- 

class DNN(nn.Module):

    def __init__(self, num_epochs, device='msp', batch_size=1, learning_rate=0.1):
        super(DNN, self).__init__()

        self.num_epochs = num_epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def _build_model(self):
        
        num_numerical_categories = len(self.num_columns)

        input_dim = num_numerical_categories
        if batch_size > 1:
            self.mlp = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.25),

                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),

                    nn.Linear(64, 1),
            )
    
    def forward(self, x_num):
        mlp_out = self.mlp(x_num)

        return mlp_out

    def fit(self, X_train, y_train, X_val, y_val):

        self.num_columns = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

        ## -- Building model --
        self._build_model()
        self.to(self.device)

        X_train_num = X_train[self.num_columns]
        X_val_num = X_val[self.num_columns]

        train_X_tensor = torch.tensor(X_train_num.values, dtype=torch.float).to(self.device)
        train_y_tensor = torch.tensor(y_train.values, dtype=torch.float).to(self.device).unsqueeze(-1)
        val_X_tensor = torch.tensor(X_val_num.values, dtype=torch.float).to(self.device)
        val_y_tensor = torch.tensor(y_val.values, dtype=torch.float).to(self.device).unsqueeze(-1)

        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            self.train() 
            
            train_loss = 0
            
            for i_step, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                prediction = self.forward(x)   
                loss_value = criterion(prediction, y)
                loss_value.backward()
                optimizer.step()
                
                train_loss += loss_value.item()

            train_loss /= len(train_loader)
            
            self.eval()
            val_loss = 0
            val_correct = 0  
            val_total = 0
            with torch.no_grad():
                for i, (X, y) in enumerate(val_loader): 
                    outputs = self.forward(X)
                    loss_value_val = criterion(outputs, y)
                    val_loss += loss_value_val.item()

                    probs = torch.sigmoid(outputs)  
                    preds = (probs > 0.5).float()
                    val_correct += (preds == y).sum().item()
                    val_total += y.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = val_correct / val_total if val_total > 0 else 0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{self.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}")

            lr_scheduler.step()

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model with Val Loss: {best_val_loss:.4f}")

             
## -- Solver --

class Solver:

    def __init__(self, config):
        self.config = config
        self.model = None

        self._model_classes = {
            "CatBoostRegressor": CatBoostRegressor,
            "XGBRegressor": XGBRegressor,
            "LGBMRegressor": LGBMRegressor,
            "Ridge": Ridge,
        }

    def _build_model_(self):
        model_name = self.config.general.selected_model

        if model_name in self.config.models.classic:
            model_cfg = self.config.models.classic[model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        class_name = model_cfg.clas
        if class_name not in self._model_classes:
            raise ValueError(f"Unknown model class: {class_name}")

        model_class = self._model_classes[class_name]

        params = dict(model_cfg.get("params", {}))

        return model_class(**params)


    def cross_validate(self, df: pd.DataFrame):

        kf = KFold(**self.config.data.kfold.params)

        scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            dataset = Dataset(self.config)
            dataset.fit(train_df)
            
            train_pr = dataset.transform(train_df)
            val_pr = dataset.transform(val_df)

            X_train = train_pr.drop(columns=[self.config.data.target_column])
            y_train = np.log1p(train_pr[self.config.data.target_column])

            X_val = val_pr.drop(columns=[self.config.data.target_column])
            y_val = np.log1p(val_pr[self.config.data.target_column])


            cat_features = [
                X_train.columns.get_loc(col)
                for col in dataset.cat_features
                if col in X_train.columns
            ]

            model = self._build_model_()
            if isinstance(model, CatBoostRegressor):
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)

            print(f"Fold {fold}: RMSE = {rmse:.5f}")

        mean_score = np.mean(scores)
        print(f"CV mean RMSE: {mean_score:.5f}")

        return mean_score


    def fit(self, df: pd.DataFrame):

        target = self.config.data.target_column

        X = df.drop(columns=[target])
        y = np.log1p(df[target])

        if self.config.general.selected_model.startswith("catboost"):
            cat_features = [
                i for i, col in enumerate(X.columns)
                if X[col].dtype == object
            ]
            self.model = self._build_model_()
            self.model.fit(X, y, cat_features=cat_features)
        else:
            self.model = self._build_model_()
            self.model.fit(X.to_numpy(), y.to_numpy())

    def predict(self, df: pd.DataFrame):

        if self.config.general.selected_model.startswith("catboost"):
            return self.model.predict(df)
        else:
            return self.model.predict(df.to_numpy())




# -- RUN -- 

def train():

    dataset = Dataset(config)
    solver = Solver(config)

    df = pd.read_csv(f"{config.paths.path_to_train_data}")
    
    solver.cross_validate(df)

    dataset.fit(df)
    train = dataset.transform(df)
    solver.fit(train)

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    pickle.dump(solver.model, open("model.pkl", "wb"))

def inference():

    solver = Solver(config)
    with open("dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    solver.model = pickle.load(open("model.pkl", "rb"))

    df = pd.read_csv(f"{config.paths.path_to_test_data}")
    submission = pd.DataFrame({
        "Id": df["Id"]
    })

    test = dataset.transform(df)
    predict_log = solver.predict(test)
    predict = np.expm1(predict_log)

    submission["SalePrice"] = predict.astype(float)
    submission.to_csv(f"{config.paths.path_to_submission}/{config.general.selected_model}.csv", index=False)

if config.general.is_train:
    train()
else:
    inference()