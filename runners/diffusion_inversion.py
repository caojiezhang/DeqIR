import os
from functions import latent_space_opt_anderson
from runners.diffusion import Diffusion
import numpy as np
import tqdm
import torch
from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from functions.denoising import forward_steps, generalized_steps
from basicsr.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
import time 
import pdb
import torchvision.utils as tvu
import wandb
from PIL import Image

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

class DiffusionInversion(Diffusion):
    
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

    # SR on latent space optimization
    def ls_sr_opt(self):

        # Do initial setup
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        # First, I need to get my data!!!
        _, dataset = get_dataset(args, config)

        # Load model in eval mode!
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
                                use_fp16=True, #TODO
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

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)
            model.cuda()

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
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
        for param in model.parameters():
            param.requires_grad = False

        if self.args.use_model_ir:
            from models.network_swinir import SwinIR as swinir_net
            # model_ir = swinir_net(upscale=4, in_chans=3, img_size=64, window_size=8,
            #                     img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            # param_key_g = 'params'
            # pretrained_swinir = torch.load("/cluster/work/cvl/jiezcao/pretrained_model/SwinIR/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")    #args.swinir_path  
            model_ir = swinir_net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'
            pretrained_swinir = torch.load("/cluster/work/cvl/jiezcao/pretrained_model/SwinIR/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth")
            model_ir.load_state_dict(pretrained_swinir[param_key_g] if param_key_g in pretrained_swinir.keys() else pretrained_swinir, strict=True)
            model_ir.to(self.device)
            model_ir = torch.nn.DataParallel(model_ir)
            model_ir.eval()
        else:
            model_ir = None
        
        # load vgg model for LPIPS
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

        B = 1
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        seq = self.get_timestep_sequence()

        # degradation
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
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
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
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv
        elif self.args.deg == 'cs_walshhadamard':
            compress_by = round(1/scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
            A, Ap = A_funcs.A, A_funcs.A_pinv

        global_time = 0
        global_min_l2_dist = 0
        # epsilon value for early stopping
        eps = 0.5
        lpips_eps = 0.1 #TODO
        # img_idx = 1   #TODO
        for _ in range(self.config.ls_opt.num_samples):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            start_epoch = 0
            
            if args.use_wandb:
                run = wandb.init(project="latent-space-opt-sr-final", reinit=True, name=f"trial-{args.seed}",
                            group=f"{config.data.dataset}-{config.data.category}-DDIM-indistr-{self.config.ls_opt.in_distr}-T{args.timesteps}-parallel-{self.config.ls_opt.use_parallel}-" +
                                f"l1-{self.args.lambda1}-l2-{self.args.lambda2}-l3-{self.args.lambda3}-lr-{config.optim.lr}-" + 
                                 f"-devices-2",
                                #f"-devices-{torch.cuda.device_count()}",
                            settings=wandb.Settings(start_method="fork"),
                            config=args
                            )
            
            if len(dataset) == 1:
                img_idx = 0 
            else:
                img_idx = 1 #TODO np.random.randint(low=0, high=len(dataset))

            x_init, _ = dataset[img_idx]
            x_target = x_init.view(1, C, H, W).float().cuda()
            x_target = data_transform(self.config, x_target)
            _, c, h, w = x_target.shape
            x_lq = A(x_target)
            # x_up = Ap(x_lq).view(-1, c, h, w)

            if model_ir is not None:
                x_up = model_ir(((x_lq+1)/2).view(-1, c, h//scale, w//scale)) * 2 - 1
                x_up = x_up.detach()
            elif args.ref_path != "":
                import cv2
                x_up = cv2.imread(args.ref_path)
                x_up = cv2.resize(x_up, (256, 256))
                x_up = x_up.astype(np.float32) / 255.
                x_up = torch.from_numpy(x_up)[None,...].permute(0,3,1,2).to(x_lq.device)
            else:
                x_up = Ap(x_lq).view(-1, c, h, w)

            if self.config.ls_opt.use_parallel:
                x = torch.randn(
                    B,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device 
                )
                # Smart initialization for faster convergence
                # This further improves results from paper by a lot!
                if self.args.smart_init:
                    with torch.no_grad():
                        all_x, _ = forward_steps(x_target, seq, model, self.betas)
                        x = all_x[-1].detach().clone()
                        # Image.fromarray((x[0].cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)).save('all_x.png')

                # This ensures that this gradient descent updates can be performed on this  
                all_xt = torch.repeat_interleave(x, self.args.timesteps+1, dim=0).to(x.device).requires_grad_() 
                
                additional_args = latent_space_opt_anderson.get_additional_lt_opt_args_sr(all_xt, seq, self.betas, x.size(0), scale, x_lq, A, Ap, self.args)

                if self.config.ls_opt.method == 'ddpm':
                    additional_args['eta'] = self.args.eta
                    if self.args.eta == 0:
                        raise ValueError("DDPM mode but eta is 0!!!")
                    all_noiset = torch.randn(
                        self.args.timesteps * B,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device 
                    ).to(x.device)
                    additional_args['all_noiset'] = all_noiset

                anderson_config_params = {
                    "m": args.m,
                    "max_iters": args.max_anderson_iters,
                    "lambda": args.lam,
                    "tol": args.tol,
                    "beta": args.anderson_beta
                }
                optimizer = get_optimizer(self.config, [all_xt])
                min_loss = float('inf')
                best_img_src = x
                min_l2_dist = float('inf')
                max_psnr = - float('inf')
                min_lpips = float('inf')

                from functions.latent_space_opt_anderson import DEQLatentSpaceOpt
                from functions.latent_space_opt_ddpm import DEQDDPMLatentSpaceOpt
                
                if self.config.ls_opt.method == 'ddpm':
                    print("Performing optimization on DDPM!!!")
                    deq_ls_opt = DEQDDPMLatentSpaceOpt()
                else:
                    deq_ls_opt = DEQLatentSpaceOpt()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for epoch in range(start_epoch, config.training.n_epochs):
                    optimizer.zero_grad()
                    
                    xt_pred = deq_ls_opt.find_source_noise_deq_sr(all_xt, x_lq,
                                                model, 
                                                additional_args, 
                                                anderson_params=anderson_config_params, 
                                                tau=args.tau, 
                                                pg_steps=args.pg_steps, 
                                                logger=None)

                    if model_ir is not None:
                        # loss_target = (xt_pred[-1] - x_up).square().sum(dim=(1, 2, 3)).mean(dim=0) #TODO
                        loss_target = loss_fn_vgg(xt_pred[-1:], x_up)   #TODO 
                        # loss_target = (A(xt_pred[-1:]) - x_lq).square().sum(dim=(1)).mean(dim=0) #TODO
                    elif args.ref_path != "":
                        loss_target = loss_fn_vgg(xt_pred[-1:], x_up)   #TODO 
                    else:
                        loss_target = (xt_pred[-1] - x_target).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    
                    loss_reg = all_xt[0].detach().square().sum()

                    loss = args.lambda1 * loss_target
                    
                    loss.backward()
                    optimizer.step()

                    # if psnr > max_psnr:
                    if loss < min_loss:
                        print("Min loss encountered!")
                        min_loss = loss
                        best_img_src = all_xt[0].detach().clone()
                        min_l2_dist = loss_target
                    
                    eps = lpips_eps  #TODO
                    log_image = loss < eps

                    if args.use_wandb and (epoch % config.training.snapshot_freq == 0 or epoch == 0 or epoch == 1 or epoch == config.training.n_epochs-1) or log_image:
                        with torch.no_grad():
                            
                            best_img_src = best_img_src.view(B, C, H, W)
                            cur_img_latent = torch.repeat_interleave(best_img_src, self.args.timesteps, dim=0).to(x.device)
                            if self.config.ls_opt.method == 'ddpm':
                                bsz, ch, h0, w0 = cur_img_latent.shape
                                m = self.args.m
                                X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                H_and = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                                y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                                sampling_args = {
                                    'all_xt': cur_img_latent,
                                    'all_noiset': all_noiset,
                                    'X': X,
                                    'F': F,
                                    'H': H_and,
                                    'y': y,
                                    'bsz': x.size(0),
                                    'm': m,
                                }
                                sampling_additional_args = self.get_additional_anderson_args_ddpm(cur_img_latent, 
                                                        xT=best_img_src, 
                                                        all_noiset=all_noiset, 
                                                        betas=self.betas, 
                                                        batch_size=x.size(0), 
                                                        eta=self.args.eta)

                                generated_image = self.sample_image(x=cur_img_latent, model=model, 
                                                        args=sampling_args,
                                                        additional_args=sampling_additional_args,
                                                        method="ddpm")
                                
                            elif self.config.ls_opt.method == 'ddim':
                                bsz, ch, h0, w0 = cur_img_latent.shape
                                m = self.args.m
                                X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                H_and = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                                y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                                sampling_args = {
                                    'all_xt': cur_img_latent,
                                    'X': X,
                                    'F': F,
                                    'H': H_and,
                                    'y': y,
                                    'bsz': x.size(0),
                                    'm': m,
                                }
                                sampling_additional_args = self.get_additional_anderson_args_sr(cur_img_latent, 
                                                        xT=best_img_src, 
                                                        betas=self.betas, 
                                                        batch_size=x.size(0), start_timesteps=1000,
                                                        scale=scale, lq=x_lq, A=A, Ap=Ap)

                                generated_image = self.sample_image(x=cur_img_latent, model=model, 
                                                        args=sampling_args,
                                                        additional_args=sampling_additional_args,
                                                        method="anderson")
                            else:
                                generated_image = self.sample_image(best_img_src.view((B, C, H, W)), model, additional_args=additional_args, method="generalized")

                            xt_pred[-1:] = generated_image.cuda()

                            logged_images = [
                                wandb.Image(x_target.detach().squeeze().view((C, H, W))),
                                wandb.Image(generated_image.detach().squeeze().view((C, H, W))),
                            ]
                            wandb.log({
                                    "all_images": logged_images
                                    })

                    # lpips
                    lpips_value = loss_fn_vgg(xt_pred[-1:], x_target).item()

                    xt_pred = inverse_data_transform(config, xt_pred)
                    target_image = inverse_data_transform(config, x_target)

                    # psnr and ssim
                    psnr = calculate_psnr_pt(xt_pred[-1:], target_image, 4).item() 
                    ssim = calculate_ssim_pt(xt_pred[-1:], target_image, 4).item()
                    
                    print(f"Epoch {epoch}/{config.training.n_epochs} PSNR {psnr:.4f} SSIM {ssim:.4f} LPIPS {lpips_value:.4f}")   #Loss {loss.item()} xT {torch.norm(all_xt[0][-1])} dist {loss_target} " +f"reg {loss_reg}"
                                    
                    
                    if args.use_wandb:
                        log_dict = {
                            # "Loss": loss.item(),
                            "PSNR": psnr,
                            "SSIM": ssim,
                            "LPIPS": lpips_value, 
                        }

                        wandb.log(log_dict)

                    if loss < eps:
                        print(f"Early stopping! Breaking out of loop at {epoch}")
                        break

                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time = start.elapsed_time(end)

                if args.use_wandb:
                    log_dict = {
                        "min L2 dist": min_l2_dist.item(),
                        "min_loss": min_loss.item(),
                        "total time": total_time
                    }
                    wandb.log(log_dict)

                for i in range(B):
                    generated_image = self.sample_image(x=best_img_src.view((B, C, H, W)), model=model, 
                                                        args=sampling_args,
                                                        additional_args=sampling_additional_args,
                                                        method="anderson")
                    generated_image = inverse_data_transform(config, generated_image)
                    tvu.save_image(
                        generated_image[i], os.path.join(args.image_folder, f"anderson-gen-{img_idx}.png")
                    )
                    x_target = inverse_data_transform(config, x_target)
                    tvu.save_image(
                        x_target, os.path.join(args.image_folder, f"anderson-target-{img_idx}.png")
                    )
            else:
                # You can start with random initialization
                # This is much difficult case but also slower
                # x = torch.randn(
                #     B,
                #     config.data.channels,
                #     config.data.image_size,
                #     config.data.image_size,
                #     device=self.device 
                # ).requires_grad_()

                # Smart initialization for faster convergence
                # This further improves results from paper by a lot!
                with torch.no_grad():
                    all_x, _ = forward_steps(x_target, seq, model, self.betas)
                    x = all_x[-1].detach().clone()
                
                x = x.requires_grad_()

                optimizer = get_optimizer(self.config, [x])
                
                min_loss = float('inf')
                best_img_src = x
                min_l2_dist = float('inf')

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                
                for epoch in range(start_epoch, config.training.n_epochs):
                    optimizer.zero_grad()

                    xs, _ = generalized_steps(x, seq, model, self.betas, logger=None, print_logs=False, eta=self.args.eta)

                    loss_target = (xs[-1] - x_target).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    loss_reg = x.detach().square().sum()
                    loss = self.args.lambda1 * loss_target

                    loss.backward()
                    optimizer.step()
                    
                    if loss < min_loss:
                        min_loss = loss
                        best_img_src = xs[-1]
                        min_l2_dist = loss_target
                    
                    log_image = loss < eps
                    if args.use_wandb and ((epoch == 0 or epoch == config.training.n_epochs-1) or log_image):
                        with torch.no_grad():
                            generated_image = self.sample_image(x.detach().view((B, C, H, W)), model, method="generalized", sample_entire_seq=False)

                            logged_images = [
                                wandb.Image(x_target.detach().squeeze().view((C, H, W))),
                                wandb.Image(generated_image.detach().squeeze().view((C, H, W)))
                            ] #+ [wandb.Image(xs[i].detach().view((C, H, W))) for i in range(0, len(xs), len(xs)//10)]
                            wandb.log({
                                    "all_images": logged_images
                                    })

                    print(f"Epoch {epoch}/{self.config.training.n_epochs} Loss {loss} xT {torch.norm(x)} dist {loss_target} reg {loss_reg}")
                    
                    if args.use_wandb:
                        log_dict = {
                            "Loss": loss.item(),
                            "max all_xt": x.max(),
                            "min all_xt": x.min(),
                            "mean all_xt": x.mean(),
                            "std all_xt": x.std(),
                            "x grad norm": x.grad.norm(),
                            "dist ||x_0 - x*||^2": loss_target.item(),
                            "reg ||x_T||^2": loss_reg.item(),
                        }
                        wandb.log(log_dict)
                    
                    if loss < eps:
                        print(f"Early stopping! Breaking out of loop at {epoch}")
                        break


                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time = start.elapsed_time(end)

                if args.use_wandb:
                    log_dict = {
                        "min L2 dist": min_l2_dist.item(),
                        "min_loss": min_loss.item(),
                        "total time": total_time
                    }
                    wandb.log(log_dict)

                for i in range(B):
                    generated_image = self.sample_image(x.detach().view((B, C, H, W)), model, method="generalized")
                    generated_image = inverse_data_transform(config, generated_image)
                    tvu.save_image(
                        generated_image[i], os.path.join(self.args.image_folder, f"seq-gen-{img_idx}.png")
                    )
                    x_target = inverse_data_transform(config, x_target)
                    tvu.save_image(
                        x_target, os.path.join(self.args.image_folder, f"seq-target-{img_idx}.png")
                    )

            print("Summary stats for anderson acceleration")
            print(f"Average time {total_time/(epoch+1)}")
            print(f"Min l2 dist {min_l2_dist}")

            if args.use_wandb:
                run.finish()

            global_time += total_time
            global_min_l2_dist += min_l2_dist
            
            print(f"Current Overall Time    : {global_time/self.config.ls_opt.num_samples}")
            print(f"Current Overall L2 dist : {min_l2_dist/self.config.ls_opt.num_samples}")
        
            torch.cuda.empty_cache()

        print(f"Overall Time    : {global_time/self.config.ls_opt.num_samples}")
        print(f"Overall L2 dist : {min_l2_dist/self.config.ls_opt.num_samples}")

