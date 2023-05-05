import os
import re
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Union, NoReturn, Optional

PARAMETERS_KEY = 'parameters'
BEST_PARAMETERS_KEY = 'best_parameters'
MODEL_FOLDER = 'model'


def get_model_parameters(model_name: str, config_name: Optional[str] = 'config.yaml') -> dict:
    config = _get_config_(config_name)
    model_parameters = config[PARAMETERS_KEY][model_name]
    return model_parameters

def save_best_parameters(model_name: str, best_parameters: dict,
                         config_name: Optional[str] = 'config.yaml') -> NoReturn:
    config = _get_config_(config_name)
    config[BEST_PARAMETERS_KEY][model_name] = best_parameters
    with open(config_name, 'w') as f:
        yaml.dump(config, f)

def _get_config_(config_name: Optional[str] = 'config.yaml') -> dict:
    with open(config_name, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data(data_folder: Optional[str] = 'data', mode: Optional[str] = 'train') -> pd.DataFrame:
    rename_columns = {'x': 'time', 'y': 'strain'}
    pattern = r'\d+\.?\d+'
    d_types_col = {
        'hardness': int,
        'temperature': int,
        'stress': float
    }

    df_list = []
    files_names = os.listdir(data_folder)
    for num, file_name in enumerate(files_names):
        data_path = os.path.join(data_folder, file_name)
        data = pd.read_csv(data_path)
        data.rename(columns=rename_columns, inplace=True)

        hardness, degrees, stress = re.findall(pattern, file_name)
        dict_df_sample = {'hardness': hardness, 'temperature': degrees,
                          'stress': stress, 'time': data.time.values,
                          'strain': data.strain.values}
        df_sample = pd.DataFrame(dict_df_sample)
        df_list.append(df_sample)

    df = pd.concat(df_list)
    df = df.astype(d_types_col)
    return df

def target_split(data: pd.DataFrame, target_col: Optional[str] = 'strain') -> tuple[np.array]:
    train_col = [col for col in data.columns if col not in target_col]
    X = data[train_col].values.astype('float32')
    y = data[target_col].values.astype('float32')
    return X, y

def save_model(model: Union, model_name: str) -> NoReturn:
    path_model = os.path.join(MODEL_FOLDER, model_name)
    with open(path_model, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_name: str) -> Union:
    path_model = os.path.join(MODEL_FOLDER, model_name)
    with open(path_model, 'rb') as f:
        model = pickle.load(f)
    return model
