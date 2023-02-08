
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
class MD_FMDOpt(SGD_FMDOpt):
    def __init__(self, params, m=1, eta_schedule = 'constant',
                 inner_optim = LSOpt, eta=1e3,
                 surr_optim_args={'lr':1.}, reset_lr_on_step=True,
                 total_steps = 1000, total_data_points=None):
        params = list(params)
        super().__init__(params, m , eta_schedule ,  inner_optim , eta,
                     surr_optim_args , reset_lr_on_step , total_steps)

    def step(self, closure, clip_grad=False):

        #======================================================
        # closure info
        # .... comments go here
        #======================================================

        # set initial step size
        self.start = time.time()
        self.state['outer_steps'] += 1

        # compute loss + grad for eta computation
        _, f_t, inner_closure = closure(call_backward=False)
        batch_size = torch.tensor(f_t.shape[0], device='cuda')
        assert f_t.max() <= 1.
        assert f_t.min() >= 0.
        assert (1-f_t).max() <= 1.
        assert (1-f_t).min() >= 0.
        # assert len(f_t.shape) == 1
        #
        # def inner_closure(f):

        # compute g_t for half steps
        dlt_dft_1 = torch.autograd.functional.jacobian(inner_closure, f_t).detach()#.reshape(f_t.shape) # n by m
        # dlt_dft_2 = torch.autograd.functional.jacobian(inner_closure, 1-f_t).detach().reshape(f_t.shape) # n by m
        # compute the half-steps
        log_f_half_1 = torch.log(f_t.detach()) - dlt_dft_1 / self.eta
        # log_f_half_2 = torch.log((1-f_t).detach()) - dlt_dft_2 / self.eta
        # construct surrogate-loss to optimize (avoids extra backward calls)
        def surrogate(call_backward=True):
            # zero out gradients
            self.zero_grad()
            # compuute new f
            _, f, inner_closure = closure(call_backward=False)
            # compute kl
            kl_div = (f * (torch.log(f) - log_f_half_1.detach()))
            # kl_div += (1-f) * (torch.log(1-f) - log_f_half_1.detach())
            # compute full surrogate
            surr = kl_div.mean()
            # do we differentiate
            if call_backward:
                surr.backward()
            # return loss
            return surr

        # make sure we take big steps
        if self.reset_lr_on_step:
            self.inner_optim.state['step_size'] = self.init_step_size

        # check improvement
        last_loss = None

        # now we take multiple steps over surrogate
        for m in range(0,self.m):

            # get loss
            current_loss = self.inner_optim.step(surrogate)

            # add in some stopping conditions
            if self.inner_optim.state['minibatch_grad_norm'] <= 1e-6:
                break

            # update internals
            self.state['inner_steps'] += 1
            self.state['grad_evals'] += 1

            # check we are improving in terms of the surrogate
            if last_loss:
                if last_loss < current_loss:
                    self.state['surrogate_increase_flag'] = 1
            else:
                last_loss = current_loss

        #
        _, f_t_new, inner_closure = closure(call_backward=False)
        print(torch.norm(f_t_new-f_t))
        self.log_info()

        # return loss
        return current_loss
