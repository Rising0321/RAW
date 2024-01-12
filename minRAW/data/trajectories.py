import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm


class Trajectories(Dataset):
    def __init__(self,
                 data_path,
                 seq_len: int,
                 epoch: int):
        now_path = data_path[epoch%int(len(data_path))]
        self.trajectories = self.init_trajectories(now_path)
        self.uids = np.load(now_path.replace("data","uid"))
        self.seq_len = seq_len
        self.size = len(self.trajectories)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.uids[index], self.trajectories[index]

    def init_trajectories(self, data_path):
        trajectories_list = np.load(data_path)
        now_data = torch.tensor(trajectories_list, dtype=float)
        return now_data
