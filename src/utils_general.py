'''
CIFAR10/100: model defintions are from https://github.com/kuangliu/pytorch-cifar/
Imagenet: from torchvision 0.13.1

The configuration for vit on cifar10/100 follows:
https://github.com/kentaroy47/vision-transformers-cifar10
'''

import random
import os
import operator as op
import matplotlib.pyplot as plt
import warnings
import torch, torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import PreActResNet18, PreActResNet50, PreActResNet101, Wide_ResNet, VGG
from models import resnet18, resnet34, resnet50, resnet101
from vit_pytorch.vit_for_small_dataset import ViT
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad
# from warmup_scheduler import GradualWarmupScheduler
from typing import List, Optional, Tuple
import timm

def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(args, device=None):

    if args.dataset.startswith('cifar'):
        num_classes = 10 if args.dataset == 'cifar10' else 100
        if args.arch == 'preactresnet18':
            model = PreActResNet18(args.gate, args.shallow, num_classes)
        elif args.arch == 'preactresnet50':
            model = PreActResNet50(args.gate, args.shallow, num_classes)
        elif args.arch == 'preactresnet101':
            model = PreActResNet101(args.gate, args.shallow, num_classes)
        # elif args.arch == 'wrn28':
            # model = Wide_ResNet(28, 10, 0.3, num_classes)
        # elif args.arch == 'vgg19':
            # model = VGG('VGG19', num_classes)
        # elif args.arch == 'vit_small':
            # model = ViT(
                # image_size=32,
                # patch_size=4,
                # num_classes=num_classes,
                # dim=512,
                # depth=6,
                # heads=8,
                # mlp_dim=512,
                # dropout=0.1,
                # emb_dropout=0.1
                        # )
        # else:
            # raise NotImplementedError("model not included")
    # else:
    else:
        if args.arch == 'simplevit':
            from vit_pytorch import SimpleViT

            model = SimpleViT(
                    image_size = 224,
                    patch_size = 16,
                    num_classes = 1000,
                    dim = 1024,
                    depth = 6,
                    heads = 16,
                    mlp_dim = 2048
            )
        elif args.arch == 'inception_v3':
            model = timm.create_model('inception_v3', pretrained=False)
        elif args.arch.startswith('vit'):
            if args.arch == 'vit_t_16':
                model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
            elif args.arch == 'vit_s_16':
                model = timm.create_model('vit_small_patch16_224', pretrained=False)
            elif args.arch == 'vit_b_16':
                model = timm.create_model('vit_base_patch16_224', pretrained=False)
        elif args.arch == 'swin_s':
            model = timm.create_model('swin_small_patch4_window7_224', pretrained=False)
        elif args.arch == 'swin_b':
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
        elif args.arch.startswith('resnet'):
            if args.arch[6:] == '18':
                model = resnet18(args.gate)
            elif args.arch[6:] == '50':
                model = resnet50(args.gate)
            elif args.arch[6:] == '101':
                model = resnet101(args.gate)
        else:
            model = torchvision.models.get_model(args.arch)

    # if args.pretrain:
        # model.load_state_dict(torch.load(args.pretrain, map_location=device))
        # model.to(device)
        # print("\n ***  pretrain model loaded: "+ args.pretrain + " *** \n")

    if device is not None:
        model.to(device)

    return model

def get_optim(parameters, args):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    if args.optim.startswith("sgd"):
        opt = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    elif args.optim == "adamw":
        opt = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == "adam":
        opt = optim.Adam(parameters, lr=args.lr)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, Adam, AdamW are supported.")

    # check if milestone is an empty array
    if args.lr_scheduler_type == "multistep":
        _milestones = [args.epoch/ 2, args.epoch * 3 / 4]
        main_lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif args.lr_scheduler_type == 'cosine':
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch - args.lr_warmup_epoch, eta_min=0.)
    elif args.lr_scheduler_type == "fixed":
        main_lr_scheduler = None
    else:
        raise ValueError('invalid lr_schduler=%s' % args.lr_scheduler_type)

    if args.lr_warmup_epoch > 0:
        if args.lr_warmup_type == 'linear':
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                    opt, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        elif args.lr_warmup_type == 'constant':
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                    opt, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epoch)
        else:
            raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epoch]
                )
    else:
        lr_scheduler = main_lr_scheduler

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

def ep2itr(epoch, loader):
    try:
        data_len = loader.dataset.data.shape[0]
    except AttributeError:
        data_len = loader.dataset.tensors[0].shape[0]
    batch_size = loader.batch_size
    iteration = epoch * np.ceil(data_len/batch_size)
    return iteration

def remove_module(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups
