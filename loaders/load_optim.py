
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
from optimizers.sgd_fmdopt import SGD_FMDOpt
from optimizers.ada_fmdopt import Ada_FMDOpt
from optimizers.diag_ada_fmdopt import Diag_Ada_FMDOpt
from optimizers.lsopt import LSOpt
from optimizers.sadagrad import Sadagrad
from optimizers.sls_fmdopt import SLS_FMDOpt
from optimizers.adam_fmdopt import Adam_FMDOpt
from optimizers.online_newton_fmdopt import Online_Newton_FMDOpt
from optimizers.gulf2 import GULF2
from optimizers.mirror_descent_fmdopt import MD_FMDOpt

# ======================
# set expensive to compute hyper-parameters
def load_train_args(args, model, loss_func, L, X, y):

    if args.algo == 'SGD':
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else (1/L)
        optim_args = {'lr': args.stepsize}
        optim = torch.optim.SGD(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
            'loss_func': loss_func, 'X':X, 'y':y, 'call_closure':True,
            'total_rounds': args.epochs, 'batch_size':args.batch_size,
            'update_lr_type': args.eta_schedule, 'single_out': True,
            'normalize_training_loss': True}

    elif args.algo == 'LSOpt':
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else args.init_step_size
        optim_args = {'lr': args.stepsize, 'c':args.c,
            'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff,
            'eta_schedule':args.eta_schedule, 'total_steps':args.total_steps}
        optim = LSOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
            'loss_func': loss_func, 'X':X, 'y':y, 'call_closure':False,
            'total_rounds': args.epochs, 'batch_size':args.batch_size,
            'update_lr_type': 'constant', 'single_out': True,
            'normalize_training_loss': True}

    elif args.algo == 'Sadagrad':
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else  1e-2
        optim_args = {'lr': args.stepsize}
        optim = Sadagrad(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
            'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': True,
            'total_rounds': args.epochs, 'batch_size':args.batch_size,
            'update_lr_type': 'constant', 'single_out': True,
            'normalize_training_loss': True}

    elif args.algo == 'MD_FMDOpt':
        assert args.loss == 'CrossEntropyLoss'
        optimal_stepsize = 1.
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule':args.eta_schedule,
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps, 'reset_lr_on_step':args.reset_lr_on_step}
        optim = MD_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False, 'include_data_id': False}

    elif args.algo == 'GULF2':
        optimal_stepsize = 1.
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'surr_optim_args':surr_optim_args,
            'prox_steps': args.gulf2_prox_steps, 'alpha':args.gulf2_alpha, 'reg_lambda':1e-2}
        optim = GULF2(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False}

    elif args.algo == 'SGD_FMDOpt':
        optimal_stepsize = 1/4 if args.loss=='MSELoss' else 2.
        optimal_stepsize = 1 / (1-1/torch.tensor(y).unique().shape[0]) if args.loss=='CrossEntropyLoss' else optimal_stepsize
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule':args.eta_schedule,
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps, 'reset_lr_on_step':args.reset_lr_on_step}
        optim = SGD_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False}

    elif args.algo == 'Ada_FMDOpt':
        optimal_stepsize = 1e2
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule': 'constant',
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps, 'reset_lr_on_step':args.reset_lr_on_step}
        optim = Ada_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False}

    elif args.algo == 'Diag_Ada_FMDOpt':
        optimal_stepsize = 1
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule': 'constant',
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps,
                      'total_data_points':y.shape[0], 'reset_lr_on_step':args.reset_lr_on_step}
        optim = Diag_Ada_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'include_data_id': True, 'normalize_training_loss': False}

    elif args.algo == 'Online_Newton_FMDOpt':
        assert args.loss == 'BCEWithLogitsLoss'
        optimal_stepsize = 1.
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule': 'constant',
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps,
                      'total_data_points':y.shape[0], 'reset_lr_on_step':args.reset_lr_on_step}
        optim = Online_Newton_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'include_data_id': True, 'normalize_training_loss': False}

    elif args.algo == 'SLS_FMDOpt':
        optimal_stepsize = 1.
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule':args.eta_schedule,
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps, 'reset_lr_on_step':args.reset_lr_on_step,
                      'c':args.outer_c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
                      'beta_update':args.outer_beta_update, 'expand_coeff':args.expand_coeff,}
        optim = SLS_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False}

    elif args.algo == 'Adam_FMDOpt':
        optimal_stepsize = 1/4 if args.loss=='MSELoss' else 2.
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else optimal_stepsize
        surr_optim_args = {'lr':args.init_step_size, 'c':args.c, 'n_batches_per_epoch': y.shape[0] / args.batch_size,
            'beta_update':args.beta_update, 'expand_coeff':args.expand_coeff, 'eta_schedule':'constant'}
        optim_args = {'eta':1/args.stepsize, 'eta_schedule':args.eta_schedule, 'total_data_points':y.shape[0],
                      'inner_optim':eval(args.inner_opt), 'surr_optim_args':surr_optim_args,
                      'm':args.m, 'total_steps':args.total_steps, 'reset_lr_on_step':args.reset_lr_on_step }
        optim = Adam_FMDOpt(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
                'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': False,
                'total_rounds': args.epochs, 'batch_size':args.batch_size,
                'update_lr_type': 'constant', 'single_out': False,
                'normalize_training_loss': False, 'include_data_id': True}

    elif args.algo == 'Adam':
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else  1e-3
        optim_args = {'lr': args.stepsize}
        optim = torch.optim.Adam(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
            'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': True,
            'total_rounds': args.epochs, 'batch_size':args.batch_size,
            'update_lr_type': 'constant', 'single_out': True,
            'normalize_training_loss': True}

    elif args.algo == 'Adagrad':
        args.stepsize = 10**args.log_lr if not args.use_optimal_stepsize else  1e-2
        optim_args = {'lr': args.stepsize}
        optim = torch.optim.Adagrad(model.parameters(), **optim_args)
        train_args = {'args':args, 'model':model, 'optim':optim,
            'loss_func': loss_func, 'X':X, 'y':y, 'call_closure': True,
            'total_rounds': args.epochs, 'batch_size':args.batch_size,
            'update_lr_type': 'constant', 'single_out': True,
            'normalize_training_loss': True}
    else:
        raise Exception()

    return optim, train_args
