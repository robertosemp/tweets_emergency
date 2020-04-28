from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch

class tweetDataset(Dataset):
    
    def __init__(self, np_X, np_Y):
        
        #np_X is a numpy array of dimension (num_examples, num_features)
        #np_Y is a numpy array of dumension (num_examples, 1)
        self.X = torch.from_numpy(np_X)
        self.Y = torch.from_numpy(np_Y)
        self.size = self.Y.shape[0]
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        
