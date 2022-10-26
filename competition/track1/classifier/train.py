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
from models import DiscreteLinearModel, ContinuousLinearModel
from loaders.load_data import load_libsvm
from optimizers.sgd_fmdopt import SGD_FMDOpt
from optimizers.ada_fmdopt import Ada_FMDOpt
from optimizers.lsopt import LSOpt
from optimizers.sadagrad import Sadagrad
from parser import *
from train import *
from helpers import get_grad_norm, get_grad_list, get_random_string, update_exp_lr, update_stoch_lr

def train_model(args, model, optim, loss_func, X, y, update_lr_type='constant', single_out=False,
            call_closure=False, total_rounds = 1000, batch_size=100, log_rate=1, include_data_id=False,
            accumulate_grad=False, normalize_training_loss=False):

    # form a data index set
    data_idxs = torch.tensor([_ for _ in range(y.shape[0])])

    # log stuff
    dataset = torch.utils.data.TensorDataset(X, y, data_idxs)
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, drop_last=True)
    logs, s, starting_time = [], 0,  time()
    import_vals = ['inner_steps', 'avg_loss', 'function_evals', 'grad_evals', 'lr',
            'step_time', 'inner_step_size', 'inner_backtracks', 'outer_stepsize',
            'eta']

    # iterate over epochs
    for t in tqdm(range(total_rounds)):

        # log everything
        if t % log_rate == 0:
            avg_loss = 0.
            grad_norm = 0.
            optim.zero_grad()
            for X_batch, y_batch, data_idx_batch in data_generator:
                # put data onto the
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                def closure(call_backward=True):
                    loss = loss_func(model(X_batch), y_batch)
                    if call_backward==True:
                        loss.backward()
                    return loss
                # add loss to average
                avg_loss += closure(call_backward=True).detach().cpu().numpy()

            # compute norm of cumulative gradient
            grad_norm = get_grad_norm(model.parameters()).detach().cpu().numpy()
            log_info = {'avg_loss': avg_loss / y.shape[0],
                        'optim_steps': s, 'function_evals': s, 'grad_evals': s,
                        'inner_backtracks': 0, 'inner_steps': 1,
                        'grad_norm': (grad_norm / torch.tensor(y.shape[0])).item(),
                        'eta_scale': args.stepsize,
                        'time_elapsed':  time() - starting_time}

            log_info.update({key:optim.state[key] for key in optim.state.keys() if key in import_vals})
            log_info.update({'function_evals+grad_evals': log_info['function_evals']+log_info['grad_evals']})
            # # log info
            try:
                wandb.log(log_info)
            except:
                raise Exception
            logs.append(log_info)
            print('=========================================================')
            print('number of Epochs:', t)
            print(log_info)
            print('=========================================================')

        # step through data by sampling without replacement
        for X_batch, y_batch, data_idx_batch in tqdm(data_generator,leave=False):

            # put data onto the
            X_batch, y_batch, data_idx_batch = X_batch.cuda(), y_batch.cuda(), data_idx_batch.cuda()

            # create closure for line-search/lbfgs
            def closure(call_backward=True, single_out=single_out):
                model_outputs = model(X_batch)
                def inner_closure(model_outputs):
                    # print(model_outputs.shape, y_batch.shape)
                    loss = loss_func(model_outputs, y_batch)
                    if normalize_training_loss:
                        loss /= data_idx_batch.shape[0]
                    return loss
                loss = inner_closure(model_outputs)
                if call_backward==True:
                    loss.backward()
                if not single_out:
                    return loss_func, X_batch, y_batch, model
                else:
                    return loss

            # if we need to call it before hand (SGD/Adam/Adagrad)
            if call_closure:
                optim.zero_grad()
                closure()

            # step optimizer over closure
            if not include_data_id:
                step_loss = optim.step(closure)
            else:
                step_loss = optim.step(closure, data_idx_batch)

            s += 1
            if update_lr_type == 'constant':
                pass
            elif update_lr_type == 'stochastic':
                optim = update_stoch_lr(optim, torch.tensor(s).float(), torch.tensor(args.stepsize).float())
            elif update_lr_type == 'exponential':
                optim = update_exp_lr(optim, torch.tensor(s).float(), torch.tensor(total_rounds*y.shape[0]/batch_size).int(), torch.tensor(args.stepsize).float())
            else:
                raise Exception

            # check for nans
            assert step_loss == step_loss
            assert grad_norm == grad_norm

        # early stopping conditions
        if (grad_norm / torch.tensor(y.shape[0])).item() < 1e-6:
            break

    # reformat stored data
    parsed_logs = {}
    for key in log_info.keys():
        try:
            parsed_logs[key] = torch.tensor([i[key] for i in logs])
        except:
            pass

    # return info
    return model, 