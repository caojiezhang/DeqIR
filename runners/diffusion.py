import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.nn.functional as F
import PIL
from PIL import Image
from models.diffusion import Model

from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
import time 
from torchvision.transforms.functional import normalize

import torchvision
import torchvision.utils as tvu
from torchvision import transforms
import wandb
import pdb


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif beta_schedule == "geometric":
        ratio = 1 - beta_end
        betas = np.array([(ratio**n) for n in range(1, num_diffusion_timesteps+1)], dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1-alphas_cumprod).sqrt()
        sigma_ks = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        lambda_ = 10 
        self.rhos = lambda_*(0.05**2) / (sigma_ks**2)

        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    
    def sample(self):
        cls_fn = None

        if self.config.model.type == 'simple':
            model = Model(self.config)  

        elif self.config.model.type == 'openai':
            from guided_diffusion.script_util import (NUM_CLASSES, model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict)
            from utils import utils_model
            model_name = '256x256_diffusion_uncond'
            ckpt = 'pretrain_model/256x256_diffusion_uncond.pt'
            model_config = dict(model_path=ckpt,
                                num_channels=128,
                                num_res_blocks=1,
                                attention_resolutions="16",
            ) if model_name == 'diffusion_ffhq_10m' \
                else dict(model_path=ckpt,
                    num_channels=256,
                    num_res_blocks=2,
                    use_fp16=True,
                    attention_resolutions="8,16,32",)
            model_args = utils_model.create_argparser(model_config).parse_args([])
            model, diffusion = create_model_and_diffusion(**args_to_dict(model_args, model_and_diffusion_defaults().keys()))
            if self.config.model.use_fp16:
                model.convert_to_fp16()

        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "ImageNet":
            name = "imagenet"
        elif self.config.data.dataset == "CelebA_HQ":
            name = "celeba_hq"
        else:
            raise ValueError
        
        if name == 'celeba_hq':
            ckpt = os.path.join("pretrain_model/celeba_hq.ckpt")
            if not os.path.exists(ckpt):
                download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
        
        print("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()
        
        self.sample_ir(model, method=self.args.method, start_timesteps=self.args.start_timesteps)


    def reset_seed(self, seed):
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def sample_ir(self, model, method='anderson', start_timesteps=1000): 
        cls_fn = None
        args, config = self.args, self.config
        dataset, test_dataset = get_dataset(args, config)
        
        val_loader = data.DataLoader(test_dataset, 
                                     batch_size=config.sampling.batch_size,
                                     shuffle=False, 
                                     num_workers=config.data.num_workers)

        scale = self.args.scale
        if self.args.deg == 'sr_averagepooling':
            if self.args.use_svd:
                from functions.svd_operators import SuperResolution
                A_funcs = SuperResolution(config.data.channels, config.data.image_size, scale, self.device)
                A = A_funcs.A
                Ap = A_funcs.A_pinv
            else:
                A = torch.nn.AdaptiveAvgPool2d((256//scale, 256//scale)) 
                Ap = lambda z: MeanUpsample(z,scale)
        elif self.args.deg == 'sr_bicubic':
            self.args.use_svd = True
            factor = scale
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                            config.data.channels, self.config.data.image_size, self.device, stride=factor)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load(f"experiments/inp_masks/{self.args.mask_type}.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            mask_tensor = torch.from_numpy(loaded)[None,:][None,:].to(self.device)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv    
        elif self.args.deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10 
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel_size = 5 
            kernel = torch.Tensor([pdf(x) for x in range(-(kernel_size // 2), kernel_size // 2 + 1)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            kernel_size = 9  
            A_funcs = Deblurring(torch.Tensor([1 / kernel_size] * kernel_size).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'cs_walshhadamard':
            deg_scale = 0.5 
            compress_by = round(1/deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv

        excel_result = []
        sum_psnr = sum_ssim = sum_lpips_val = 0
        total_time = 0
        img_id = 200 if self.config.data.dataset == 'CelebA_HQ' else 0  
        
        pbar = tqdm.tqdm(val_loader)
        with torch.no_grad(): 
            for gt_x, classes in pbar:

                n = config.sampling.batch_size
                _, c, h, w = gt_x.shape

                # gt image
                gt_x = gt_x.to(self.device)
                gt_x = data_transform(self.config, gt_x)
                gt_patch = gt_x
                
                # lq image
                in_patch = A(gt_patch)
                
                # add noise
                if self.args.add_noise:
                    in_patch = in_patch + torch.randn_like(in_patch).cuda() * self.args.sigma_y
                
                if config.sampling.batch_size!=1:
                    raise ValueError("please change the config file to set batch size as 1")

                # initiliation 
                if self.args.init_type == 'noise':
                    x = torch.randn(
                        1,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if method == 'anderson':
                    all_xt = torch.repeat_interleave(x, self.args.timesteps, dim=0).to(x.device)
                    bsz, ch, h0, w0 = all_xt.shape
                    m = self.args.m
                    X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    H = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                    y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                    args = {
                        'all_xt': all_xt,
                        'X': X,
                        'F': F,
                        'H': H,
                        'y': y,
                        'bsz': x.size(0),
                        'm': m,
                    }
                    additional_args = self.get_additional_anderson_args_sr(all_xt, xT=x, betas=self.betas, batch_size=x.size(0), \
                                                        start_timesteps=start_timesteps, scale=scale, lq=in_patch, A=A, Ap=Ap)
                    
                    sample_start = time.time()
                    x = self.sample_image(x, model, args=args, additional_args=additional_args, method=method, max_iter=self.args.max_anderson_iters)
                
                else:
                    x = self.sample_image(x, model, method=method)
                    x = x.cpu()
                
                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)

                if type(x) == dict:
                    x_transformed = {}
                    for t in x.keys():
                        cur_img_idx = img_id
                        x_transformed[t] = inverse_data_transform(config, x[t])
                        for i in range(n):
                            tvu.save_image(
                                x_transformed[i], os.path.join(self.args.image_folder, str(t), "{cur_img_idx}.png")
                            )
                            cur_img_idx += 1
                else:
                    x = inverse_data_transform(config, x)
                    img_name = f"{str(img_id).zfill(5)}.png" if (self.config.data.dataset == 'CelebA_HQ') else classes[0][:-4]+'png'

                    # save image
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, img_name)   
                        )
                img_id += 1


    def get_timestep_sequence(self):
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.timesteps
            seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        elif self.args.skip_type == 'custom':
            seq = [s for s in range(0, 500, 2)] + [s for s in range(499, 1000, 4)] 
        else:
            raise NotImplementedError
        return seq
    

    def get_timestep_sequence_sr(self, start_timesteps=1000):
        assert start_timesteps <= self.num_timesteps
        assert start_timesteps >= self.args.timesteps

        if self.args.skip_type == "uniform":
            skip = start_timesteps // self.args.timesteps
            seq = range(0, start_timesteps, skip)
            if start_timesteps % self.args.timesteps != 0:
                seq = list(seq)[:-1]
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(start_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        elif self.args.skip_type == 'custom':
            seq = [s for s in range(0, 500, 2)] + [s for s in range(499, 1000, 4)] 
        else:
            raise NotImplementedError
        return seq

    def get_entire_timestep_sequence(self):
        seq = range(0, self.num_timesteps)
        return seq

    def sample_image(self, x, model, method, args=None, additional_args=None, last=True, sample_entire_seq=False, max_iter=15):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.sample_type == "generalized":
            from functions.ddim_anderson import fp_implicit_iters_anderson

            logger=None
            if method == 'anderson':
                use_wandb = False
                if use_wandb:
                    wandb.init( project="DEQ-Efficiency-exp-rebuttal",
                                group=f"DDIM-xt-init-{self.config.data.dataset}-{self.config.data.category}-{self.args.method}-device{torch.cuda.device_count()}-{len(self.betas)}-{additional_args['T']}-m-{args['m']}-steps-15",
                                reinit=True,
                                config=self.config)
                    logger = wandb.log
                
                xs = fp_implicit_iters_anderson(x, model, self.betas, args=args, 
                               additional_args=additional_args, logger=logger, print_logs=True, max_iter=max_iter) #[1, 3, 32, 32]
                if True or type(xs[0]) == dict:
                    xs = xs[0]
                    last = False
            x = xs
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x


    def get_additional_anderson_args_sr(self, all_xt, xT, betas, batch_size, start_timesteps, scale, lq, A, Ap):
        
        from functions.ddim_anderson import compute_alpha
        
        seq = self.get_timestep_sequence_sr(start_timesteps)   
        cur_seq = list(seq)                 
        seq_next = [-1] + list(seq[:-1])    
        gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]    
        xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]   
        next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]  
        prev_idx = [idx + 1 for idx in next_idx]    

        plot_timesteps = []
        
        T = len(cur_seq)    
        t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)        
        next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)  
        
        at = compute_alpha(betas, t.long())             
        at_next = compute_alpha(betas, next_t.long())   

        alpha_ratio = (at_next/at[0]).sqrt()            
        all_xT = torch.repeat_interleave(xT, T, dim=0).to(all_xt.device) 
        if self.args.use_svd:
            all_xT = alpha_ratio * (all_xT - Ap(A(all_xT.reshape(all_xT.size(0), -1))).reshape(*all_xT.size()))
        else:
            all_xT = alpha_ratio * (all_xT - Ap(A(all_xT))) 

        et_prevsum_coeff = at_next.sqrt()   

        sigma_t = (1 - at_next**2).sqrt().to(all_xt.device) 
        
        nw = torch.tensor(self.args.nw).expand(T, 1, 1, 1).to(all_xt.device)  
        etw = torch.tensor(self.args.etw).expand(T, 1, 1, 1).to(all_xt.device)  

        c1 = (1 - at_next).sqrt() * nw
        c2 = (1 - at_next).sqrt() * (1 - etw**2).sqrt() 
        c3 = (((1 - at)*at_next)/at).sqrt()

        et_coeff1 = (1 / at_next.sqrt()) * c3
        if self.args.use_svd:
            et_coeff2 = (1 / at_next.sqrt()) * c2  
        else:
            et_coeff2 = (1 / at_next.sqrt()) * c2 * sigma_t  
        
        all_y = torch.repeat_interleave(lq, T, dim=0).to(all_xt.device)
        y_coeff = at_next.sqrt() 
        if self.args.use_svd:
            all_y = y_coeff * Ap(all_y.reshape(all_y.size(0), -1)).reshape(*all_xT.size()) 
        else:
            all_y = y_coeff * Ap(all_y) 

        if self.args.use_svd:
            noise_coeff = (1 / at_next.sqrt()) * c1 
        else:
            noise_coeff = (1 / at_next.sqrt()) * c1 * sigma_t
        
        all_noise = torch.repeat_interleave(torch.randn_like(xT), T, dim=0).to(all_xt.device)  

        additional_args = {
            "all_xT": all_xT, 
            "et_coeff1": et_coeff1,
            "et_coeff2": et_coeff2,
            "et_prevsum_coeff": et_prevsum_coeff, 
            "T" : T, 
            "t" : t,
            "bz": batch_size,
            "plot_timesteps": plot_timesteps,
            "gather_idx": gather_idx,
            "xT_idx": xT_idx,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "xT": xT,
            "A": A,
            "Ap": Ap,
            'sf': scale,
            'y_coeff': y_coeff,
            'all_y': all_y,
            'noise_coeff': noise_coeff,
            'all_noise': all_noise,
            'use_svd': self.args.use_svd
        }
        return additional_args
