# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
import math


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)


#
#
# class AdaX(Optimizer):
#     r"""Implements AdaX algorithm.
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 1e-4))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-12)
#         weight_decay (float, optional): L2 penalty (default: 5e-4)
#     .. _Adam\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """
#
#     def __init__(self, params, lr=1.5e-3, betas=(0.9, 1e-4), eps=1e-12,
#                  weight_decay=5e-4):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay)
#         super(AdaX, self).__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super(AdaX, self).__setstate__(state)
#
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError('AdaX does not support sparse gradients, please consider SparseAdam instead')
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#
#                 beta1, beta2 = group['betas']
#                 state['step'] += 1
#
#                 if group['weight_decay'] != 0:
#                     grad.add_(group['weight_decay'], p.data)
#                 t = state['step']
#                 # Decay the first and second moment running average coefficient
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                 exp_avg_sq.mul_(1 + beta2).addcmul_(beta2, grad, grad)
#
#                 denom = exp_avg_sq.sqrt().add_(group['eps'])
#
#                 bias_correction2 = ((1 + beta2) ** state['step'] - 1)
#
#                 step_size = group['lr'] * math.sqrt(bias_correction2)
#                 # step_size = group['lr']
#                 p.data.addcdiv_(-step_size, exp_avg, denom)
#
#
#         return loss
#
#
#
# class AdaXW(Optimizer):
#     r"""Implements Adam algorithm.
#     It has been proposed in `AdaX: Adaptive Gradient Descent with Exponential Long Term Memory`_.
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 1e-4))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay (default: 5e-2
#     .. _Adam\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """
#
#     def __init__(self, params, lr=0.005, betas=(0.9, 1e-4), eps=1e-12,
#                  weight_decay=5e-2):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay)
#         super(AdaXW, self).__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super(AdaXW, self).__setstate__(state)
#
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#
#
#         for group in self.param_groups:
#
#             beta1, beta2 = group['betas']
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 if grad.is_sparse:
#                     raise RuntimeError('AdaX does not support sparse gradients, please consider SparseAdam instead')
#
#
#                 state = self.state[p]
#
#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     # Exponential moving average of gradient values
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     # Exponential moving average of squared gradient values
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)
#
#
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#
#
#                 state['step'] += 1
#
#                 exp_avg.mul_(beta1).add_(1 - beta1, grad)
#                 exp_avg_sq.mul_(1 + beta2).addcmul_(beta2, grad, grad)
#
#                 denom = exp_avg_sq.sqrt().add_(group['eps'])
#
#                 bias_correction2 = ((1 + beta2) ** state['step'] - 1)
#
#                 step_size = group['lr'] * math.sqrt(bias_correction2)
#
#                 p.data.add_(-torch.mul(p.data, group['lr'] * group['weight_decay'])).addcdiv_(-step_size, exp_avg,
#                                                                                               denom)
#
#         return loss