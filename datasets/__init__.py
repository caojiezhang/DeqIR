import os
import torch
import numbers
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
from functools import partial
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        if args.no_augmentation: 
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            tran_transform = transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CIFAR10Pair":
        dataset = CIFAR10Pair(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
            fixed_noise=config.data.fixed_noise,
        )
        test_dataset = CIFAR10Pair(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
            fixed_noise=config.data.fixed_noise
        )

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "CelebA_HQ" or config.data.dataset == 'FFHQ':
        if config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "ood_celeba"),
                transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                              transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, "datasets", "CelebA-HQ"), 
                transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                              transforms.ToTensor()])
            )
            num_items = len(dataset)
            indices = list(range(num_items))
            if not config.data.subset_100:
                random_state = np.random.get_state()
                np.random.set_state(random_state)
            train_indices, test_indices = (
                indices[: int(num_items * 0.)],
                indices[int(num_items * 0.) :],
            )
            if config.data.subset_100:
                test_indices = test_indices[::10]
            test_dataset = Subset(dataset, test_indices)

    elif config.data.dataset == "demo":
        if args.no_resize:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, args.path_test), 
                transform=transforms.Compose([transforms.ToTensor()])
            )
        else:
            dataset = torchvision.datasets.ImageFolder(
                    os.path.join(args.exp, args.path_test), 
                    transform=transforms.Compose([transforms.Resize([config.data.image_size, config.data.image_size]),
                                                transforms.ToTensor()])
                )
        num_items = len(dataset)
        indices = list(range(num_items))
        test_dataset = Subset(dataset, indices)

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == 'ImageNet':
        # only use validation dataset here
        
        if config.data.subset_1k:
            from datasets.imagenet_subset import ImageDataset
            if  config.data.subset_100:
                sub_name = 'imagenet_val_100'
            else:
                sub_name = 'imagenet_val_1k'

            dataset = ImageDataset(os.path.join('experiments', 'datasets', 'imagenet', 'imagenet_val_1k'),   #args.exp,  small_
                     os.path.join('experiments', 'datasets', 'imagenet', sub_name+'.txt'),  #args.exp,  small_
                     image_size=config.data.image_size,
                     normalize=False)
            test_dataset = dataset
        
        elif config.data.out_of_dist:
            dataset = torchvision.datasets.ImageFolder(
                os.path.join(args.exp, 'datasets', 'ood'),
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
        else:
            dataset = torchvision.datasets.ImageNet(
                os.path.join(args.exp, 'datasets', 'imagenet'), split='val',
                transform=transforms.Compose([partial(center_crop_arr, image_size=config.data.image_size),
                transforms.ToTensor()])
            )
            test_dataset = dataset
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
