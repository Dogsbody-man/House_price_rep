from omegaconf import OmegaConf

config = {
    'general': {
        'is_train': True,  
        'experiment_name': 'default',
        'task': 'regression', 
        'seed': 0xFACED,
        'selected_model': 'dnn'
    },
    'paths': {
        'path_to_train_data': './data/train.csv',
        'path_to_test_data': './data/test.csv',
        'path_to_sample_submission': './data/sample_submission.csv',
        'path_to_submission': './results',
        'path_to_scaler': './scaler'
    },
    'data': {
        'id_column': 'Id',
        'target_column': 'SalePrice', 

        'train_batch_size': '${data.batch_size}',
        'val_batch_size': '${data.batch_size}',

        'kfold': {
            'name': 'KFold',
            'params': {
                'n_splits': 5,
                'shuffle': True
            }
        },

        'train_size': 0.8,
        'val_size': 0.2,
    },
    'models': {  
        'dnn': {    
            'name': 'Net',
            'params': {
                'batch_size': 64, 
                'num_epochs': 200,
                'lr': 1e-5, 
                'early_stopping_epochs': 10,
                'device': 'mps'
            }
        },
        'classic': {
            'xgboost': {
                'clas': 'XGBRegressor',
                'params': {
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'learning_rate': 0.05,
                    'max_depth': 4,
                    'min_child_weight': 3,
                    'n_estimators': 400,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'subsample': 0.9,
                    'tree_method': 'hist',
                }
            },
            'ridge': {
                'clas': 'Ridge',
                'params': {
                    'alpha': 10,
                }
            },
            'catboost': {
                'clas': 'CatBoostRegressor',
                'params': {
                    'eval_metric': '${metric.catboost}',
                    'random_seed': '${general.seed}', 
                    'verbose': 200, 
                    'early_stopping_rounds': 200, 
                    'depth': 4, 
                    'iterations': 6000, 
                    'l2_leaf_reg': 1, 
                    'learning_rate': 0.005,
                }
            },
            'catboost_num':{
                'clas': 'CatBoostRegressor',
                'params': {
                    'eval_metric': '${metric.catboost}',
                    'random_seed': '${general.seed}', 
                    'verbose': 200, 
                    'early_stopping_rounds': 200, 
                    'depth': 4, 
                    'iterations': 6000, 
                    'l2_leaf_reg': 1, 
                    'learning_rate': 0.005,
                }
            },
            'lgbm': {
                'clas': 'LGBMRegressor',
                'params': {
                    'max_depth': 3, 
                    'min_child_samples': 10, 
                    'n_estimators': 300, 
                    'n_jobs': -1,
                    'num_leaves': 130, 
                    'random_seed': 42, 
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.1, 
                    'subsample': 0.6,
                }
            }
        }
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
            'weight_decay': 1e-5,
        }
    },
    'scheduler': {
        'name': 'ReduceLROnPlateau', 
        'params': {
            'mode': 'min',
            'patience': 10, 
            'factor': 0.5, 
            'verbose': True,
        }
    },
    'loss': {
        'name': 'MSELoss'
    },
    'metric': {
        'catboost': 'RMSE',
        'sklearn_name': 'neg_root_mean_squared_error',
    },
}

config = OmegaConf.create(config)