# imports
import torch
import numpy as np
from copy import deepcopy
import time

# local imports
from helpers import *
from optimizers.lsopt import LSOpt

# optimizer
class Sadagrad(torch.optim.Optimizer):
    def __init__(self, params, eta_schedule = 'constant',
                 lr=1e-2):
        params = list(params)
        super().__init__(params, {})

        # create some local tools
        self.params = params
        self.grad_sum = None
        self.lr = lr

    @staticmethod
    def squared_grad_norm(params):
        g_list = []
        for p in params:
            grad = p.grad
            if grad is None:
                grad = 0.
            g_list += [grad]
        grad_norm = 0.
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        # assert 1==0
        for g in g_list:
            if g is None:
                continue
            if torch.sum(torch.mul(g, g)).device != device:
                grad_norm += torch.sum(torch.mul(g, g)).to(device)
            else:
                grad_norm += torch.sum(torch.mul(g, g))
        return grad_norm

    @staticmethod
    def get_grad_list(params):
        g_list = []
        for p in params:
            grad = p.grad
            if grad is None:
                grad = 0.
            g_list += [grad]
        return g_list

    def step(self, closure, clip_grad=False):

        # update grad-norm
        if self.grad_sum:
            self.grad_sum += self.squared_grad_norm(self.params)
        else:
            self.grad_sum = self.squared_grad_norm(self.params)

        # print('sadagrad', self.lr / ( self.grad_sum).pow(0.5))
        # save the current parameters:
        params_current = deepcopy(self.params)
        grad_current = deepcopy(get_grad_list(self.params))

        # only do the check if the gradient norm is big enough
        with torch.no_grad():

            # =================================================
            # apply rescaling of the step-size
            for p_next, p_current, g_current in zip(self.params, params_current, grad_current):
                p_next.data = p_current - self.lr * g_current / (self.grad_sum).pow(0.5)

        # return loss
        return closure(call_backward=False)
