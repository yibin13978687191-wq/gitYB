"""Tabular data preprocessing for ANN-style models."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class DataPreprocessor:
    def __init__(self, x_dataframe, y_dataframe, test_size=0.2):
        if not 0 <= test_size < 1:
            raise ValueError("test_size must be in [0, 1).")

        x_array = x_dataframe.to_numpy(dtype=np.float32)
        y_array = y_dataframe.to_numpy(dtype=np.float32)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)

        if test_size == 0:
            self.x_train, self.y_train = x_array, y_array
            self.x_val = self.y_val = None
        else:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                x_array,
                y_array,
                test_size=test_size,
                random_state=42,
            )

        self.x_train = torch.from_numpy(self.x_train)
        self.y_train = torch.from_numpy(self.y_train)
        if self.x_val is not None:
            self.x_val = torch.from_numpy(self.x_val)
            self.y_val = torch.from_numpy(self.y_val)

    def create_dataloaders(self, batch_size=8, shuffle=True):
        train_dataset = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        if self.x_val is None:
            return train_loader

        val_dataset = TensorDataset(self.x_val, self.y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_input_dim(self):
        return self.x_train.shape[1]

    def get_output_dim(self):
        return self.y_train.shape[1]
