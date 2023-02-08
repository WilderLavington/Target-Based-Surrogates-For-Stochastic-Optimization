
# imports
import torch
import numpy as np
from copy import deepcopy
import time

# local imports
from helpers import *
from optimizers.lsopt import LSOpt

from helpers import get_grad_norm, get_grad_list, get_random_string, update_exp_lr, update_stoch_lr

# helpers

# linesearch optimizer
class SGD_FMDOpt(torch.optim.Optimizer):
    def __init__(self, params, m=1, eta_schedule = 'constant',
                 inner_optim = LSOpt, eta=1e3,
                 surr_optim_args={'lr':1.}, reset_lr_on_step=True,
                 total_steps = 1000, include_rel_reg=False):
        params = list(params)
        super().__init__(params, {})

        # create some local tools
        self.params = params
        self.m = m
        self.total_steps = total_steps
        self.include_rel_reg = include_rel_reg
        self.surr_optim_args = surr_optim_args
        # set eta and the divergence
        # self.inner_optim = inner_optim(self.params,**surr_optim_args)
        self.div_op = lambda f, ft: torch.norm(f-ft,2).pow(2)
        self.eta = eta # please rename
        self.eta_schedule = eta_schedule
        self.inner_lr = surr_optim_args['lr']
        self.reset_lr_on_step = reset_lr_on_step

        # preset eta (parameter-wise / diagnol only)
        # for inner_opt in self.inner_optim.param_groups[0]['params']:
        #     inner_opt.data = inner_opt.data.to('cuda')
        self.init_step_size = 10
        # store state for debugging
        self.state['outer_steps'] = 0
        self.state['inner_steps'] = 0
        self.state['inner_backtracks'] = 0
        self.state['step_time'] = None
        self.state['inner_step_size'] = None
        self.state['grad_evals'] = 0
        self.state['function_evals'] = 0
        self.state['outer_step_size'] = None
        self.state['surrogate_increase_flag'] = 0

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

    def log_info(self):
        # try logging (generalized for different inner-optimizers )
        try:
            assert isinstance(self.inner_optim.state['function_evals'], int)
            self.state['function_evals'] = self.inner_optim.state['function_evals']
            self.state['inner_step_size'] = self.inner_optim.state['step_size']
            self.state['outer_stepsize'] = 1/eta
        except:
            self.state['function_evals'] += 1
            self.state['inner_step_size'] = self.inner_lr
        self.state['step_time'] = timer(self.start,time.time())
        return None

    def zero_grad(self):
        self.inner_optim = LSOpt(self.params,**self.surr_optim_args)
        self.inner_optim.zero_grad()

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
