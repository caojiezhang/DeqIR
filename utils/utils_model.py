# -*- coding: utf-8 -*-
import numpy as np
import torch
from functools import partial

from guided_diffusion.script_util import add_dict_to_argparser
import argparse
import pdb


# ----------------------------------------
# wrap diffusion model
# ----------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def model_fn(x, noise_level, model_diffusion, vec_t=None, model_out_type='pred_xstart', \
        diffusion=None, ddim_sample=False, alphas_cumprod=None, **model_kwargs):

    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    # time step corresponding to noise level
    if not torch.is_tensor(vec_t):
        t_step = find_nearest(reduced_alpha_cumprod,(noise_level/255.))
        vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)
        # timesteps = torch.linspace(1, 1e-3, num_train_timesteps, device=device)
        # t = timesteps[t_step]
    if not ddim_sample:
        out = diffusion.p_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
        )
    else:
        out = diffusion.ddim_sample(
            model_diffusion,
            x,
            vec_t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            eta=0,
        )

    if model_out_type == 'pred_x_prev_and_start':
        return out["sample"], out["pred_xstart"]
    elif model_out_type == 'pred_x_prev':
        return out["sample"]
    elif model_out_type == 'pred_xstart':
        return out["pred_xstart"]
    elif model_out_type == 'epsilon':
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        return out
    elif model_out_type == 'score':
        alpha_prod_t = alphas_cumprod[int(t_step)]
        beta_prod_t = 1 - alpha_prod_t
        out = (x - alpha_prod_t ** (0.5) * out["pred_xstart"]) / beta_prod_t ** (0.5)
        out = - out / beta_prod_t ** (0.5)
        return out


'''
# ---------------------------------------
# print
# ---------------------------------------
'''


# -------------------
# print model
# -------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# -------------------
# print params
# -------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


'''
# ---------------------------------------
# information
# ---------------------------------------
'''


# -------------------
# model inforation
# -------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# -------------------
# params inforation
# -------------------
def info_params(model):
    msg = describe_params(model)
    return msg


'''
# ---------------------------------------
# description
# ---------------------------------------
'''


# ----------------------------------------------
# model name and total number of parameters
# ----------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# ----------------------------------------------
# parameters description
# ----------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), name) + '\n'
    return msg

# ----------------------------------------
# load model
# ----------------------------------------

def create_argparser(model_config):
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path='',
        diffusion_steps=1000,
        noise_schedule='linear',
        num_head_channels=64,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,
        num_heads=4,
        num_heads_upsample=-1,
        use_new_attention_order=False,
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        channel_mult="",
        learn_sigma=True,
        class_cond=False,
        use_checkpoint=False,
        image_size=256,
        num_channels=128,
        num_res_blocks=1,
        attention_resolutions="16",
        dropout=0.1,
    )
    defaults.update(model_config)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def grad_and_value(operator, x, x_hat, measurement, weights=[1/3,1/3,1/3]):
    
    if not isinstance(operator, list):
        operator = [operator]

    if not isinstance(x_hat, list):
        x_hat = [x_hat]

    if not isinstance(measurement, list):
        measurement = [measurement]

    x_op = []
    for op in operator:
        for x_h in x_hat:
            x_op.append(op(x_h))
    
    norm = 0
    # difference = []
    for i in range(len(x_hat)):
        difference = measurement[i] - x_op[i]
        norm += torch.linalg.norm(difference) * weights[i]
    import pdb; pdb.set_trace()
    # norm = norm / len(x_hat)
    norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
    return norm_grad, norm

