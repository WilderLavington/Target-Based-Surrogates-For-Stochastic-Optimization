
# general imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import wandb
import torch.nn as nn
import numpy as np
from time import time
import pathlib
from torch.optim import SGD, Adam, Adagrad
import os

# get local imports
from models import DiscreteLinearModel, ContinuousLinearModel, LinearNueralNetworkModel
from loaders.load_exp import load_model, load_dataset

from parser import *
from train import *
from loaders.load_optim import *
from helpers import get_grad_norm, get_grad_list, get_random_string, update_exp_lr, update_stoch_lr

def check_args(args):
    # make sure configs match for mfac work
    if args.dataset_name=='mfac':
        assert args.loss=='MSELoss'
    # check optim configs
    if args.algo in ['Sadagrad', 'Ada_FMDOpt', 'Diag_Ada_FMDOpt', 'Adam', 'Adagrad']:
        assert args.eta_schedule == 'constant'
    # make sure inner-opt is good.
    if args.algo in ['Ada_FMDOpt', 'SGD_FMDOpt', 'Diag_Ada_FMDOpt']:
        assert args.inner_opt =='LSOpt'
    # make sure inner-opt is good.
    if args.fullbatch:
        assert args.batch_size == 100.
    if args.algo == 'Online_Newton_FMDOpt':
        assert args.loss in ['BCEWithLogitsLoss', 'CrossEntropyLoss']
    if args.algo == 'OMD_FMDOpt':
        assert args.loss in ['BCELoss', 'NLLLoss']
    # make sure we are matching the problem correctly
    if args.dataset_name in ['mnist', 'cifar10', 'cifar100']:
        assert args.loss == 'CrossEntropyLoss'

def main():

    # get arguments
    args, parser = get_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # verify args work
    check_args(args)

    # initialize weights and biases runs
    log_path = 'wandb_logs/'+args.dataset_name+'/'+args.algo+'/'+args.loss
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    # If you don't want your script to sync to the cloud
    os.environ['WANDB_DIR'] = log_path
    os.environ['WANDB_MODE'] = 'online'
    wandb.init(project=args.project, entity=args.entity, config=args,
                group=args.group) #, dir=args.log_dir

    # local logging
    pathlib.Path('logs/'+args.folder_name).mkdir(parents=True, exist_ok=True)


    # get dataset  and model
    X, y = load_dataset(data_set_id=args.dataset_name, data_dir='datasets/', loss=args.loss)
    model, loss_func, L = load_model(data_set_id=args.dataset_name, loss=args.loss, X=X, y=y)

    # set "optimal stepsize"
    args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else (1/L)

    # set batch size to fullbatch for gradient decent
    args.batch_size = y.shape[0] if args.fullbatch else args.batch_size

    # to account for batch-size (e.g. make sure we take more steps with bigger batches)
    if args.normalize_epochs_lengths:
        args.m = 1 if args.algo in ['SGD', 'LSOpt', 'Adam', 'Adagrad', 'Sadagrad'] else args.m
        args.epochs = max(int(args.episodes * (1 / args.m) * (args.batch_size / y.shape[0])), args.min_epochs)
    else:
        args.epochs = max(int(args.episodes * (args.batch_size / y.shape[0])), args.min_epochs)
        assert self.eta_schedule != 'exponential'

    # compute total steps given the # of epochs
    args.total_steps = int(args.epochs * (y.shape[0] / args.batch_size))

    # get optimizer and the training args
    optim, train_args = load_train_args(args, model, loss_func, L, X, y)

    # call model training script
    model, logs = train_model(**train_args)

    # store logs
    if args.randomize_folder:
        file=get_random_string(16)
        try:
            os.makedirs('logs/database/'+file)
        except FileExistsError:
            print("File already exists")
        torch.save(logs, 'logs/database/'+file+'/'+args.file_name+'.pt')
        torch.save(args, 'logs/database/'+file+'/args.pt')
        torch.save(model, 'logs/database/'+file+'/model.pt')
    else:
        torch.save(logs, 'logs/'+args.folder_name+args.file_name+'.pt')
        torch.save(args, 'logs/'+args.folder_name+'args.pt')
        torch.save(args, 'logs/'+args.folder_name+'model.pt')

if __name__ == "__main__":
    main()
