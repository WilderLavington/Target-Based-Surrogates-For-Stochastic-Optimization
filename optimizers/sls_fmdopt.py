
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
class SLS_FMDOpt(SGD_FMDOpt):
    def __init__(self, params, m=1, eta_schedule = 'constant',
                 inner_optim = LSOpt, eta=1e3,
                 surr_optim_args={'lr':1.}, reset_lr_on_step=True,
                 total_steps = None, n_batches_per_epoch=None,
                 c=0.5, beta_update=0.9, expand_coeff=1.8):
        params = list(params)
        super().__init__(params, m , eta_schedule ,  inner_optim , eta,
                     surr_optim_args , reset_lr_on_step , total_steps )
        # create some local tools
        self.grad_sum = None
        self.init_eta = eta
        self.c = c
        self.beta_update = beta_update
        self.expand_coeff = expand_coeff
        self.min_eta = 1e-4
        self.max_eta = 1e4

    def compute_functional_stepsize(self, inner_closure, f_t, dlt_dft):
        eta_prop = self.eta / self.expand_coeff
        for i in range(100):
            lhs = inner_closure(f_t - (1/eta_prop) * dlt_dft)
            rhs = inner_closure(f_t) - (1/eta_prop) * self.c * torch.norm(dlt_dft).pow(2)
            if lhs > rhs:
                eta_prop /= self.beta_update
            elif (1/eta_prop) <= 1e-6:
                break
            else:
                break
        return eta_prop

    def step(self, closure, clip_grad=False):

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

        # solve for eta + project it to make sure it does not explode
        self.eta = self.compute_functional_stepsize(inner_closure, target_t, dlt_dft)
        self.eta = max(self.min_eta, self.eta)
        self.eta = min(self.max_eta, self.eta)

        # set  eta schedule
        if self.eta_schedule == 'constant':
            eta = self.eta
        elif self.eta_schedule == 'stochastic':
            eta = self.eta * torch.sqrt(torch.tensor(self.state['outer_steps']).float())
        elif self.eta_schedule == 'exponential':
            eta = self.eta * torch.tensor((1/self.total_steps)**(-self.state['outer_steps']/self.total_steps)).float()
        else:
            raise Exception

        # construct surrogate-loss to optimize (avoids extra backward calls)
        def surrogate(call_backward=True):
            #
            self.inner_optim.zero_grad()
            # f = n by m
            target = model(X_t)
            # m by d -> 1
            loss = dlt_dft*target
            # remove cap F
            reg_term = (target - target_t.detach()).pow(2)
            # compute full surrogate
            surr = (loss / eta + reg_term ).mean()
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
