import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from tensorboardX import SummaryWriter

from runners.diffusion import Diffusion
from runners.diffusion_inversion import DiffusionInversion
import pdb

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=666, help="Random seed")   
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--out_name", type=str, default="outputs", help="Path for saving running related data.")
    parser.add_argument("--n_subset", type=str, default="100", help="100 | 1k")
    parser.add_argument("--path_test", type=str, default="", help="Path of the test dataset.",)
    parser.add_argument("--doc", type=str, required=True, help="A string for documentation purpose. ""Will be the name of the log folder.",)
    parser.add_argument("--comment", type=str, default="", help="A string for experiment comment")
    parser.add_argument("--verbose", type=str, default="info", help="Verbose level: info | debug | warning | critical",)
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model",)
    parser.add_argument("-i", "--image_folder", type=str, default="",  help="The folder name of samples",)
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher",)
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--sample_type", type=str, default="generalized", help="sampling approach (generalized or ddpm_noisy)",)
    parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)",)
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--start_timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma",)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--model", type=str, default="Diffusion", help="model to use to train -- Diffusion or DiffusionInversion",)
    parser.add_argument("--method", type=str, default="anderson", help="sampling method to use anderson",)
    parser.add_argument("--ls_sr_opt", action="store_true", help="When true, performs sr latent space optimization",)
    parser.add_argument("--lambda1", type=float, default=1, help="Value of regularization parameter on ||X_pred - x^**||",)
    parser.add_argument("--lambda2", type=float, default=0, help="Value of regularization parameter on x_T",)
    parser.add_argument("--lambda3", type=float, default=0, help="Value of regularization parameter for fp",)
    parser.add_argument("--tau", type=float, default=0.1, help="Value of damping parameter in anderson",)
    parser.add_argument("--pg_steps", type=int, default=1, help="Number of steps while computing phantom gradients",)
    parser.add_argument("--use_wandb", action="store_true", help="Value of damping parameter in anderson",)
    parser.add_argument("--no_augmentation", action="store_true", help="If set to true, no augmenttion will be applied to dataset",)
    parser.add_argument("--m", type=int, default=5, help="Anderson parameters m",)
    parser.add_argument("--max_anderson_iters", type=int, default=15, help="Max number of iterations for Anderson", )
    parser.add_argument("--lam", type=float, default=0.001, help="Anderson parameter lambda",)
    parser.add_argument("--tol", type=float, default=0.01, help="Tolerance for residuals in Anderson acceleration", )
    parser.add_argument("--anderson_beta", type=float, default=1.0, help="Anderson parameter beta",)
    parser.add_argument("--init_type", type=str, default="noise", help="initiliation type", )
    parser.add_argument("--deg", type=str, default="sr_bicubic", help="sr_averagepooling | sr_bicubic",)
    parser.add_argument("--mask_type", type=str, default="", help="mask type in image inpainting",)
    parser.add_argument("--use_svd", action="store_true", help="use svd or not", )
    parser.add_argument("--nw", type=float, default=0, help="the weight to control c1 and c2") 
    parser.add_argument("--etw", type=float, default=0, help="the weight to control c1 and c2") 
    parser.add_argument("--scale", type=int, default=1, help="scale in sr",)
    parser.add_argument("--test_real", action="store_true", help="test real image or not",)
    parser.add_argument("--no_resize", action="store_true", help="When true, not resize images",)
    parser.add_argument("--smart_init", action="store_true", help="When true, use smart initilation",)
    parser.add_argument("--ref_path", type=str, default="", help="Path of reference iamges.")
    parser.add_argument("--add_noise", action="store_true", help="When true, add noise",)
    parser.add_argument("--sigma_y", type=float, default=0.2, help="noise level",)
    parser.add_argument("--use_model_ir", action="store_true", help="use model_ir or not",)

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.sample and not args.ls_sr_opt:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                shutil.rmtree(tb_path)
                os.makedirs(args.log_path)
                if os.path.exists(tb_path):
                    shutil.rmtree(tb_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = SummaryWriter(log_dir=tb_path)   #TODO
        # new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))
        
        handler1 = logging.StreamHandler()
        if not os.path.exists(args.log_path):
            os.makedirs(args.log_path)
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        new_config.tb_logger = SummaryWriter(log_dir=tb_path)   

        data_name = args.config.split('.')[0] + args.n_subset 
        if 'demo' in data_name:
            data_name = os.path.join(data_name + "_" + args.doc)

        if args.scale > 4:
            task_name = f'{args.method}_timesteps{args.timesteps}_eta{args.eta}_svd{args.use_svd}_nw{args.nw}_etw{args.etw}_scale{args.scale}'    #_seed{args.seed}
        else:
            task_name = f'{args.method}_timesteps{args.timesteps}_eta{args.eta}_svd{args.use_svd}_nw{args.nw}_etw{args.etw}' #_seed{args.seed}
        
        if 'sr' in args.deg:
            data_name = os.path.join(data_name, 'x'+str(args.scale))
        
        out_name = args.out_name 
        os.makedirs(os.path.join(args.exp, out_name, args.deg, data_name), exist_ok=True)
        if args.mask_type != "":
            task_name = task_name + '_' + args.mask_type

        args.image_folder = os.path.join(
            args.exp, out_name, args.deg, data_name, task_name+args.image_folder  
        )
        
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input(
                    f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                )
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.image_folder)
                os.makedirs(args.image_folder)
            else:
                print("Output image folder exists. Program halted.")
                sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = eval(args.model)(args, config)
        if args.sample:
            print("Performing image restoration!!!")
            runner.sample()
        elif args.ls_sr_opt:
            print("Performing initialization optimization!!!")
            runner.ls_sr_opt()
        else:
            NotImplementedError("Sample type is not defined")
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
