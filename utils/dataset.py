import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, version):
        self.X = X
        self.y = y
        self.version = version

    def call(self):
        print(f"Dataset Version is {self.version} !!!")
        return 0
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype = torch.float32), torch.tensor(self.y[idx], dtype = torch.long)
    
    def __len__(self):
        return len(self.X)