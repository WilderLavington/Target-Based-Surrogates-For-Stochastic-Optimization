
# imports
import torch
import numpy as np
from copy import deepcopy
import time

# local imports
from helpers import *
from optimizers.lsopt import LSOpt

from helpers import get_grad_norm, get_grad_list, get_random_string, update_exp_lr, update_stoch_lr
from optimizers.lsopt import LSOpt

# helpers

# linesearch optimizer
class GULF2(torch.optim.Optimizer):
    def __init__(self, params, reg_lambda=1e-5,
                 surr_optim_args={'lr':1.},
                 prox_steps = 5, alpha=0.1):
        params = list(params)
        super().__init__(params, {})

        # create some local tools
        self.params = params
        self.params_t = deepcopy(params)
        self.prox_steps = prox_steps
        self.current_step = 0
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.surr_optim_args = surr_optim_args
        # set eta and the divergence
        # self.inner_optim = torch.optim.Adam(self.params, **surr_optim_args)

        # store state for debugging
        self.state['outer_steps'] = 0
        self.state['inner_steps'] = 0
        self.state['grad_evals'] = 0
        self.state['function_evals'] = 0

    @staticmethod
    def compute_grad_norm(grad_list):
        grad_norm = 0.
        grad_list = [g.reshape(-1) for g in grad_list]
        flat_grad = torch.cat(grad_list, dim=0).reshape(-1)
        grad_norm = torch.sqrt(flat_grad.pow(2).sum())
        return grad_norm
    @staticmethod
    def get_grad_list(params):
        g_list = []
        for p in params:
            grad = p.grad
            if grad is None:
                grad = torch.tensor( 0., device='cuda')
            g_list += [grad.reshape(-1)]
        return g_list
    @staticmethod
    def copy_params(target, source):
        """ copies nueral network parameters between to networks. """
        for target_param, param in zip(target, source):
            target_param.data.copy_(param.data)

    def step(self, closure, clip_grad=False):

        # set initial step size
        self.start = time.time()
        self.state['outer_steps'] += 1
        self.state['surrogate_increase_flag'] = 0
        self.current_step += 1

        # check if we are updating f_t
        if self.current_step == self.prox_steps:
            self.copy_params(self.params_t, self.params)

        # compute wierd prox term
        current_params = deepcopy(self.params)
        self.copy_params(self.params, self.params_t)
        loss_func, X_t, y_t, model = closure(call_backward=False)

        #
        def inner_closure(model_outputs):
            loss = loss_func(model_outputs, y_t)
            return loss
        self.inner_optim = LSOpt(model.parameters(),**self.surr_optim_args)
        target_t = model(X_t)

        # produce some 1 by m (n=batch-size, m=output of f)
        self.inner_optim.zero_grad()
        dlt_dft = torch.autograd.functional.jacobian(inner_closure, target_t).detach() # n by m

        # put original params back
        self.copy_params(self.params, current_params)
        batch_size = torch.tensor(target_t.shape[0], device='cuda')

        # construct surrogate-loss to optimize (avoids extra backward calls)
        def surrogate(call_backward=True):
            #
            self.zero_grad()
            # f = n by m
            loss_func, X_t, y_t, model = closure(call_backward=False)
            target = model(X_t)
            # m by d -> 1
            prox_term = torch.sum(dlt_dft*target)
            # compute full surrogate
            surr = (loss_func(model(X_t), y_t) + (1-self.alpha) * prox_term) / batch_size
            #
            R = torch.norm(torch.nn.utils.parameters_to_vector(self.params), 2).pow(2)
            # do we differentiate
            if call_backward:
                (surr + self.reg_lambda * R).backward()
            # return loss
            return surr + self.reg_lambda * R

        # compute the current loss
        current_loss = self.inner_optim.step(surrogate)

        # update internals
        self.state['inner_steps'] += 1
        self.state['grad_evals'] += 1
        self.state['function_evals'] += 1

        # return loss
        return current_loss
