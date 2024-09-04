import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from itertools import product
from copy import copy, deepcopy
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
import random
import time
from functools import partial

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def rMAE(pred, true):
    return np.sum(np.abs(true-pred)) / np.sum(np.abs(true))

def rRMSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2) / np.sum(true**2))

@dataclass
class Mesh:
    full : torch.Tensor
    part : list

def meshify(l, device, requires_grad = False):
    parts = [torch.tensor(part, requires_grad = requires_grad, dtype=torch.get_default_dtype()).to(device) for part in np.array_split(np.array(list(product(*l))), 2, 1)]
    full = torch.cat(parts, dim=1)
    return Mesh(full, parts)

def generate_mesh_object(point_counts, domain, skew = None, device = 'cpu', full_requires_grad = True, border_requires_grad = False):
    if isinstance(point_counts, int):
        point_counts = [point_counts]
        domain = [domain]

    if skew:
        full_list = [np.linspace(0, 1, count) for i, count in enumerate(point_counts)]
        full_list = [points ** skew[i] for i, points in enumerate(full_list)]
        full_list = [domain[i][0] + (domain[i][1] - domain[i][0]) * points for i,points in enumerate(full_list)]
    else:
        full_list = [np.linspace(domain[i][0], domain[i][1], count) for i, count in enumerate(point_counts)]
    
    full_mesh = meshify(full_list, device, full_requires_grad)
    
    border_lists = [(copy(full_list), copy(full_list)) for _ in full_list]
    for i, _ in enumerate(full_list):
        border_lists[i][0][i] = [float(domain[i][0])]
        border_lists[i][1][i] = [float(domain[i][1])]
    border_meshes = [(meshify(border[0], device, border_requires_grad), meshify(border[1], device, border_requires_grad)) for border in border_lists]
    
    return full_mesh, border_meshes

def f(model, mesh, of = None):
    points = mesh
    if isinstance(mesh, Mesh):
        points = mesh.full
    if of is not None:
        return torch.split(model(points), 1, 1)[of]
    return model(points)

def df(model, mesh, of = 0, wrt = 0, order = 1):
    df_of = f(model, mesh, of)
    respect_to = mesh.part[wrt]
    gradoutputs = torch.ones_like(respect_to)
    for _ in range(order):
        df_of = torch.autograd.grad(
            df_of,
            respect_to,
            grad_outputs=gradoutputs,
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_of
 

def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    
    b_left = data[0,:,:] 
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)

    return res, b_left, b_right, b_upper, b_lower


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def make_temporal_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src

def make_time_sequence(src, num_step=5, step=1e-4):
    return make_temporal_sequence(src, num_step, step)


# TODO implement the spatial sequence
def make_spatial_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_data_3d(x_range, y_range, t_range, x_num, y_num, t_num):
    step_x = (x_range[1] - x_range[0]) / float(x_num-1)
    step_y = (y_range[1] - y_range[0]) / float(y_num-1)
    step_t = (t_range[1] - t_range[0]) / float(t_num-1)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]

    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1)
    res = data.reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[0]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_left = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[1]:x_range[1]+step_x:step_x,y_range[0]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_right = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[0]:y_range[0]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_lower = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    x_mesh, y_mesh, t_mesh = np.mgrid[x_range[0]:x_range[1]+step_x:step_x,y_range[1]:y_range[1]+step_y:step_y,t_range[0]:t_range[1]+step_t:step_t]
    b_upper = np.squeeze(np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(y_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1))[1:-1].reshape(-1,3)

    return res, b_left, b_right, b_upper, b_lower