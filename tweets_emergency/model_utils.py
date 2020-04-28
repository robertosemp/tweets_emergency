import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd

def predict(model, X):
    
    '''
    model - a pytorch model object
    X - data input as a pandas dataframe in the shape (n, k) where n is the number of 
    examples and k is the number of features
    '''
 
    torch_tensor = torch.tensor(X.values)    #Converting the pandas dataframe into a pytorch tensor
    return pd.DataFrame(model.forward(torch_tensor.float()))