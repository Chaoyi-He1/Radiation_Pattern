import os
import scipy.io as sio
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import random


class Pointnet_Dataset(Dataset):
    def __init__(self, data_folder_path: str, is_train: bool = True, 
                 num_points: int = 361, cache: bool = False,
                 split: float = 0.8):
        self.data_folder_path = data_folder_path
        self.data_files_path = [os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.csv')]
        self.data_files_path.sort()
        self.is_train = is_train
        self.split = split * len(self.data_files_path)
        self.data_files_path = self.data_files_path[:int(self.split)] if is_train else self.data_files_path[int(self.split):]
        
        self.num_points = num_points

        self.data = []
        self.cache = cache
        if cache:
            self.load_data()
    
    def load_data(self):
        for f in self.data_files_path:
            data = pd.read_csv(f, header=None).to_numpy()
            self.data.append(data)
    
    def __len__(self) -> int:
        return len(self.data_files_path)
    
    def __getitem__(self, index):
        if self.cache:
            data = self.data[index]
        else:
            data = pd.read_csv(self.data_files_path[index], header=None).to_numpy()
        
        # Randomly selecting one point from every 10 points in data
        indices = [random.randint(i*10, min((i+1)*10 - 1, data.shape[0]-1)) for i in range(data.shape[0] // 10)]
        data_sampled = data[indices]
        
        return data_sampled
    
    def collate_fn(self, batch):
        data_sampled = list(zip(*batch))
        data_sampled = torch.tensor(np.array(data_sampled)).float()
        return data_sampled
        