from typing import Tuple, List, Union, Callable, Optional, NoReturn

import pandas as pd
import numpy as np
import logging
from itertools import product
from sklearn.model_selection import KFold

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

SEED = 123
N_FEATURES = 4
OUTPUT_SIZE = 1
BATCH_SIZE = 256
EARLYSTOPPING_PATIENCE = 5
MAX_EPOCHS = 1_000
VERBOSE = True


class Net(nn.Module):
    """Torch class object of fully connected neural network."""
    def __init__(self, hidden_size: int, n_layers: int, batch_norm: bool) -> NoReturn:
        super().__init__()
        self.hidden_size = hidden_size

        layers = []
        layers.append(nn.Linear(N_FEATURES, hidden_size))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        for i in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, OUTPUT_SIZE))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return output


class LSTM(nn.Module):
    """Torch class object of LSTM."""
    def __init__(self, hidden_size: int, n_layers: int) -> NoReturn:
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out)
        return predictions


class LightningNet(pl.LightningModule):
    """PytorhLightning class object of fully connected neural network."""
    def __init__(self, model_type: str, lr: float, **kwargs) -> NoReturn:
        super().__init__()
        models_dict = {
            'NN': Net,
            'LSTM': LSTM,
        }
        self.model = models_dict[model_type](**kwargs)
        self.lr = lr

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss_function(self, predict: Tensor, target: Tensor) -> Tensor:
        target = torch.reshape(target, (-1, 1))
        return nn.functional.mse_loss(predict, target)

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Tensor:
        x, y = train_batch
        predict = self.forward(x)
        loss = self.loss_function(predict, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> NoReturn:
        x, y = val_batch
        prediction = self.forward(x)
        loss = self.loss_function(prediction, y)
        self.log('val_loss', loss)

    def test_step(self, val_batch: Tensor, batch_idx: int) -> Tensor:
        x, y = val_batch
        prediction = self.forward(x)
        loss = self.loss_function(prediction, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self) -> torch.optim:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer


class LoaderDataset(Dataset):
    """Data class object for for working with torch.utils.data.DataLoader."""
    def __init__(self, data, target_col: Optional[str] = 'strain') -> NoReturn:
        train_col = [col for col in data.columns if col not in target_col]
        self.len_data = len(data)
        self.X_data = data[train_col].values.astype('float32')
        self.Y_data = data[target_col].values.astype('float32')

    def __len__(self) -> int:
        return self.len_data

    def __getitem__(self, idx) -> Tuple[np.array]:
        return (self.X_data[idx], self.Y_data[idx])


class GridSearch:
    """Class object of search for hyperparameters on the grid with KFold."""
    def __init__(self, model_type: str, model: Union, model_parameters: dict, cv: int,
                 loss: Callable, verbose: bool) -> NoReturn:
        self.model_type = model_type
        self.model = model
        self.cv = cv
        self.loss = loss
        self.verbose = verbose

        self.logger = logging.getLogger(GridSearch.__name__)
        self.logger.setLevel(logging.INFO)

        params_names = model_parameters.keys()
        product_vals = list(product(*model_parameters.values()))
        self.product_parameters = [dict(zip(params_names, value)) for value in product_vals]

        self.best_model = None
        self.best_parameters = None
        self.best_score = float('inf')

    def fit(self, data: pd.DataFrame) -> NoReturn:
        n_iter = len(self.product_parameters)
        for iter, params in enumerate(self.product_parameters, start=1):
            score = self._params_score(data, params)
            self._update_params(score, params)
            if self.verbose:
                self.logger.info(f'{iter}/{n_iter}, score: {score}')

        dataset = LoaderDataset(data)
        self.best_model = self._fit(dataset, self.best_parameters)

    def _params_score(self, data: pd.DataFrame, params: dict) -> Union:
        folds_data = self._split_data(data)
        scores_list = []
        for train_dataset, test_dataset in folds_data:
            model = self._fit(train_dataset, params)
            fold_score = self._estimate_model(model, test_dataset)
            scores_list.append(fold_score)
        score = np.mean(scores_list)
        return score

    def _split_data(self, data: pd.DataFrame) -> List[tuple]:
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=SEED)
        folds_data = []
        for train_idx, test_idx in kf.split(data):
            train_dataset = LoaderDataset(data.iloc[train_idx])
            test_dataset = LoaderDataset(data.iloc[test_idx])
            folds_data.append([train_dataset, test_dataset])
        return folds_data

    def _fit(self, dataset: LoaderDataset, params: dict) -> Union:
        torch.manual_seed(SEED)
        len_dataset = len(dataset)
        train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(dataset, batch_size=len_dataset, shuffle=False, drop_last=False)
        model = self.model(model_type=self.model_type, **params)

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=EARLYSTOPPING_PATIENCE, mode='min')
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stop_callback], log_every_n_steps=1)
        trainer.fit(model, train_dataloader, test_dataloader)
        return model

    def _estimate_model(self, model: Union, dataset: LoaderDataset) -> float:
        x_tensor = torch.tensor(dataset.X_data)
        predict = model(x_tensor).detach().numpy().reshape(-1)
        label = dataset.Y_data
        score = self.loss(label, predict)
        return score

    def _update_params(self, score: float, params: dict) -> NoReturn:
        if score < self.best_score:
            self.best_score = score
            self.best_parameters = params
