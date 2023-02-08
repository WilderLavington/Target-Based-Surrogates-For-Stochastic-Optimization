
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import urllib
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import torch

# local imports
from loaders.load_data import *
from models import *


# ======================
# set expensive to compute hyper-parameters
L_MAP = {'mushrooms': torch.tensor(87057.2422, device='cuda'),
         'ijcnn': torch.tensor(3476.3210, device='cuda'),
         'rcv1': torch.tensor(166.4695, device='cuda'),
         'mfac0': torch.tensor(1000, device='cuda'),
         'mfac1': torch.tensor(1000, device='cuda'),
         'mfac4': torch.tensor(1000, device='cuda'),
         'mfac': torch.tensor(1000, device='cuda'),
         'mnist': torch.tensor(1000, device='cuda'),
         'cifar10': torch.tensor(1000, device='cuda'),
         'cifar100': torch.tensor(1000, device='cuda')}

# ======================
# general data-loader
def load_dataset(data_set_id, data_dir, loss):
    if data_set_id in LIBSVM_DOWNLOAD_FN.keys():
        return procces_data(*load_libsvm(data_set_id, datadir=data_dir), loss, data_set_id)
    elif data_set_id == "cifar100":
        return procces_data(*load_cifar100(data_dir), loss, data_set_id)
    elif data_set_id == "cifar10":
        return procces_data(*load_cifar10(data_dir), loss, data_set_id)
    elif data_set_id == "mnist":
        return procces_data(*load_mnist(data_dir), loss, data_set_id)
    elif data_set_id in ["mfac", "mfac1", "mfac4", "mfac0"]:
        assert loss == 'MSELoss'
        return procces_data(*generate_synthetic_mfac(), loss, data_set_id)
    else:
        raise Exception('provide correct dataset id')

# ======================
# format datasets
def procces_data(X, y, loss, data_set_id):
    if loss == 'CrossEntropyLoss':
        X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.long)
    elif loss == 'NLLLoss':
        X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.long)
    elif loss == 'BCEWithLogitsLoss':
        X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.float)
    elif loss == 'BCELoss':
        X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.float)
    elif loss == 'MSELoss':
        if data_set_id in ["mfac", "mfac1", "mfac4", "mfac0"]:
            X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.float)
        else:
            X, y = torch.tensor(X,device='cpu',dtype=torch.float), torch.tensor(y,device='cpu',dtype=torch.float).unsqueeze(1)
    else:
        raise Exception('')
    return X, y

# ======================
# get models
def load_model(data_set_id, loss, X, y, use_dense=False):

    # general loss
    if loss == 'CrossEntropyLoss':
        loss_func = nn.CrossEntropyLoss(reduction='sum')
        model = DiscreteLinearModel(X.shape[1], y.max()+1)
        model.to('cuda')
        L = L_MAP[data_set_id] * 2 * (1 - 1 / torch.unique(y).shape[0])

    elif loss == 'NLLLoss':
        loss_func = nn.NLLLoss(reduction='sum')
        model = SoftmaxDiscreteLinearModel(X.shape[1], y.max()+1)
        model.to('cuda')
        L = L_MAP[data_set_id] * 2 * (1 - 1 / torch.unique(y).shape[0])

    elif loss == 'BCEWithLogitsLoss':
        loss_func_ = nn.BCEWithLogitsLoss(reduction='sum')
        loss_func = lambda t, y: loss_func_(t.reshape(-1), y.reshape(-1))
        model = DiscreteLinearModel(X.shape[1], 1)
        model.to('cuda')
        L = L_MAP[data_set_id] * 4

    elif loss == 'BCELoss':
        loss_func_ = nn.BCELoss(reduction='sum')
        loss_func = lambda t, y: loss_func_(t.reshape(-1), y.reshape(-1))
        model = SoftmaxDiscreteLinearModel(X.shape[1], 1)
        model.to('cuda')
        L = L_MAP[data_set_id] * 4

    elif loss == 'MSELoss':
        loss_func = nn.MSELoss(reduction='sum')
        model = ContinuousLinearModel(X.shape[1], 1)
        model.to('cuda')
        L = L_MAP[data_set_id] * 2

    # update model for stuff that needs it
    if data_set_id == 'mfac':
        model = LinearNueralNetworkModel(X.shape[1], [10], 10, bias=False)
        model.to('cuda')
        L = L_MAP['mfac']
    elif data_set_id == 'mfac1':
        model = LinearNueralNetworkModel(X.shape[1], [1], 10, bias=False)
        model.to('cuda')
        L = L_MAP['mfac']
    elif data_set_id == 'mfac4':
        model = LinearNueralNetworkModel(X.shape[1], [4], 10, bias=False)
        model.to('cuda')
        L = L_MAP['mfac']
    elif data_set_id == 'mfac0':
        model = LinearNueralNetworkModel(X.shape[1], [], 10, bias=False)
        model.to('cuda')
        L = L_MAP['mfac']
    elif data_set_id == 'mnist':
        model = Mlp(n_classes=10, dropout=False)
        model.to('cuda')
    elif data_set_id == 'cifar10':
        if use_dense:
            model = DenseNet121(num_classes=10)
        else:
            model = ResNet([3, 4, 6, 3], num_classes=10)
        model.to('cuda')
    elif data_set_id == 'cifar100':
        if use_dense:
            model = DenseNet121(num_classes=100)
        else:
            model = ResNet([3, 4, 6, 3], num_classes=100)
        model.to('cuda')

    # rescale for safety
    L = 2*L

    # return it all
    return model, loss_func, L
