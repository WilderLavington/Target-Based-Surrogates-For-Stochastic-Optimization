# imports
import torch
import numpy as np
from copy import deepcopy
import time

# local imports
from helpers import *
from optimizers.lsopt import LSOpt
from optimizers.sgd_fmdopt import SGD_FMDOpt

# linesearch optimizer
class Diag_Ada_FMDOpt(SGD_FMDOpt):
    def __init__(self, params, m=1, eta_schedule = 'constant',
                 inner_optim = LSOpt, eta=1e3,
                 surr_optim_args={'lr':1.}, reset_lr_on_step=True,
                 total_steps = 1000, total_data_points=None):
        params = list(params)
        super().__init__(params, m , eta_schedule ,  inner_optim , eta,
                     surr_optim_args , reset_lr_on_step , total_steps )
        # set total_data points
        self.total_data_points = total_data_points
        self.dual_coord = None

    def step(self, closure, data_idxs, clip_grad=False):

        # set initial step size
        self.start = time.time()
        self.state['outer_steps'] += 1
        self.state['surrogate_increase_flag'] = 0

        # compute loss + grad for eta computation
        loss_func, X_t, y_t, model = closure(call_backward=False)
        self.inner_optim = LSOpt(model.parameters(),**self.surr_optim_args)

        #
        def inner_closure(model_outputs):
            loss = loss_func(model_outputs, y_t)
            return loss
        target_t = model(X_t)

        # produce some 1 by m (n=batch-size, m=output of f)
        self.inner_optim.zero_grad()
        dlt_dft = torch.autograd.functional.jacobian(inner_closure, target_t).detach() # n by m

        # initialize dual coordinate scaling
        if self.dual_coord is None:
            if len(target_t.shape)==1:
                self.dual_coord = torch.zeros((self.total_data_points,1)).cuda().detach()
            else:
                self.dual_coord = torch.zeros((self.total_data_points,*target_t.detach().shape[1:])).cuda().detach()

        # produce some 1 by m (n=batch-size, m=output of f)
        dlt_dft = torch.autograd.functional.jacobian(inner_closure, target_t).detach() # n by m

        # update dual coords
        self.dual_coord[data_idxs,:] += dlt_dft.pow(2).detach()
        self.dual_coord = torch.clamp(self.dual_coord, min=1e-5, max=1e5)

        # construct surrogate-loss to optimize (avoids extra backward calls)
        def surrogate(call_backward=True):
            #
            self.inner_optim.zero_grad()
            # f = n by m
            target = model(X_t)
            # m by d -> 1
            loss = dlt_dft * target
            # remove cap F
            reg_term = (target - target_t.detach()).pow(2) * (self.eta * self.dual_coord[data_idxs].pow(0.5))
            # compute full surrogate
            surr = loss.mean() + reg_term.mean()
            # do we differentiate
            if call_backward:
                surr.backward()
            # return loss
            return surr

        # check improvement
        last_loss = None

        # make sure we take big steps
        if self.reset_lr_on_step:
            self.inner_optim.state['step_size'] = self.init_step_size

        #
        for m in range(0,self.m):

            # compute the current loss
            current_loss = self.inner_optim.step(surrogate)

            # add in some stopping conditions
            if 'minibatch_grad_norm' in self.inner_optim.state.keys():
                if self.inner_optim.state['minibatch_grad_norm'] <= 1e-6:
                    break

            # update internals
            self.state['inner_steps'] += 1
            self.state['grad_evals'] += 1

            # check we are improving in terms of the surrogate
            if last_loss:

                if last_loss < current_loss:
                    self.state['surrogate_increase_flag'] = 1
                    # assert (last_loss > current_loss)
                last_loss = current_loss

            else:
                last_loss = current_loss

        #
        self.log_info()

        # return loss
        return current_loss
