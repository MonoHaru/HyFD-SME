import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from scipy.stats import skew, kurtosis
from utils.data_utils import timeseries_to_DataFrame


def signal_to_statistic_transform(data):
    data_list = []
    MEAN = np.round(np.mean(data), 3)
    MIN = np.min(data)
    MAX = np.max(data)
    STD = np.std(data)
    median = np.median(data)
    skewness = skew(data)
    kurt = kurtosis(data)
    a, b = np.percentile(data, q = [25,75])
    p2p = MAX - MIN
    rms = np.sqrt(np.mean(np.square(data)))
    var = np.var(data)
    
    data_list.extend([MEAN, STD, MIN, MAX, skewness, kurt, median, a, b, p2p, rms, var])
    return data_list


def calculate_snr(df, noise, i):
    signal_power = np.mean( df.loc[df['label'] == i, 'MEAN TEMP'] ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def add_noise(df, SNR, i):
    P_signal = np.mean( df.loc[df['label'] == i, 'MEAN TEMP'] ** 2)
    snr_linear = 10 ** (SNR / 10)
    P_noise = P_signal / snr_linear
    noise_std = np.sqrt(P_noise)
    
    # Add Gaussian Noise
    noise = np.random.normal(0, noise_std,  df.loc[df['label'] == i, 'MEAN TEMP'].shape)
    noisy_temp =  df.loc[df['label'] == i, 'MEAN TEMP'] + noise
 
    print(f'>>> Noise SNR db: {calculate_snr(df, noise, i)}')

    return noisy_temp, noise


def load_dataset(random_state, path, noise = False):
    df = pd.read_csv(path)

    # Add Noise 
    if noise != None:
        for i in range(6):
            noise_temp, noise_signal = add_noise(df, noise, i)
            df.loc[df['label'] == i, 'MEAN TEMP'] = noise_temp
    
    df = timeseries_to_DataFrame(df, 64, 16)
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)
    
    X_train_tdf = []
    X_test_tdf = []
    
    for idx in range(len(X_train)):
        data = signal_to_statistic_transform(X_train[idx])
        X_train_tdf.append(data)
        
    for idx in range(len(X_test)):
        data = signal_to_statistic_transform(X_test[idx])
        X_test_tdf.append(data)  
    
    X_train_tdf = np.array(X_train_tdf)
    X_test_tdf = np.array(X_test_tdf)

    X_train = np.concatenate([X_train_tdf, X_train], axis = 1)
    X_test = np.concatenate([X_test_tdf, X_test], axis = 1)
    
    return X_train, X_test, y_train, y_test