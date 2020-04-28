import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class neural(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)

    
    def predict(self, X):
    
        '''
        model - a pytorch model object
        X - data input as a pandas dataframe in the shape (n, k) where n is the number of 
        examples and k is the number of features
        '''

        torch_tensor = torch.tensor(X.values)    #Converting the pandas dataframe into a pytorch tensor
        return pd.DataFrame(self.forward(torch_tensor.float())).astype(float)