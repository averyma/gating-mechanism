import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attacks import pgd, pgd_linbp
from src.context import ctx_noparamgrad_and_eval
import ipdb
from tqdm import trange
from autoattack import AutoAttack
import numpy as np
from src.utils_log import Summary, AverageMeter, ProgressMeter
import time
from torch.utils.data import Subset
import torchattacks

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_gate(gates, pred, target):
    maxidx = torch.max(gates, dim=1)[1]
    correct = pred[range(len(maxidx)), :, maxidx].argmax(dim=1) == target
    acc1 = correct.sum()/target.size(0)*100.
    acc5 = 1.

    return acc1, acc5

def return_gate_ratio(gates):
    maxidx = torch.max(gates, dim=1)[1]
    gate_ratio = torch.zeros(gates.size(1))
    for i in range(gates.size(1)):
        gate_ratio[i] += (maxidx.cpu().numpy() == i).sum()
    gate_ratio.mul_(100/gates.size(0))
    return gate_ratio

def return_qualified(p_0, p_1, p_adv_0, p_adv_1, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        _, pred_0 = p_0.topk(1, 1, True, True)
        _, pred_1 = p_1.topk(1, 1, True, True)
        _, pred_adv_0 = p_adv_0.topk(1, 1, True, True)
        _, pred_adv_1 = p_adv_1.topk(1, 1, True, True)

        pred_0 = pred_0.t()
        pred_1 = pred_1.t()
        pred_adv_0 = pred_adv_0.t()
        pred_adv_1 = pred_adv_1.t()

        correct_0 = pred_0.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        correct_1 = pred_1.eq(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_0 = pred_adv_0.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        incorrect_1 = pred_adv_1.ne(target.view(1, -1).expand_as(pred_0)).squeeze()
        qualified = correct_0.eq(correct_1).eq(incorrect_0).eq(incorrect_1)

        return qualified

def validate(val_loader, model, criterion, args, is_main_task, whitebox=False):
    if whitebox:
        if args.dataset == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif args.dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif args.dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        # param = {'ord': np.inf,
              # 'epsilon': args.pgd_eps,
              # 'alpha': args.pgd_alpha,
              # 'num_iter': args.pgd_itr,
              # 'restarts': 1,
              # 'rand_init': True,
              # 'clip': True,
              # 'loss_fn': nn.CrossEntropyLoss(),
              # 'dataset': args.dataset}
        # param['num_iter'] = 1 if args.debug else args.pgd_itr
        # attacker = pgd(**param)
        atk = torchattacks.PGD(
            model,
            eps=args.pgd_eps,
            alpha=args.pgd_alpha,
            steps=1 if args.debug else args.pgd_itr,
            random_start=True)
        atk.set_normalization_used(mean=mean, std=std)

    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if whitebox:
                with ctx_noparamgrad_and_eval(model):
                    # delta = attacker.generate(model, images, target)
                    delta = atk(images, target) - images
            else:
                delta = 0

            # compute output
            with torch.no_grad():
                output = model(images+delta)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_task:
                progress.display(i + 1)
            if args.debug:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg

def validate_gate(val_loader, model, criterion, args, is_main_task):

    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, (images, target) in enumerate(loader):
            i = base_progress + i
            if args.gpu is not None and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.backends.mps.is_available():
                images = images.to('mps')
                target = target.to('mps')
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.no_grad():
                gates, logits = model(images)

                pred = F.softmax(logits, dim=1)
                onehotlabels_new = torch.zeros_like(pred)
                onehotlabels_new[range(len(target)), target, :] = 1
                decision_from_gate = (-onehotlabels_new*pred).sum(dim=1).argmin(dim=1)

                logits_reshape = logits.permute(2, 0, 1).reshape(-1, logits.size(1))
                target_repeat = target.unsqueeze(0).expand(logits.size(2), logits.size(0)).flatten()
                loss = (criterion(gates, decision_from_gate).mean() +
                        criterion(logits_reshape, target_repeat).mean())

            # measure accuracy and record loss
            acc1, acc5 = accuracy_gate(gates, pred, target)
            gate_ratio = return_gate_ratio(gates)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            gate1.update(gate_ratio[0], images.size(0))
            gate2.update(gate_ratio[1], images.size(0))
            gate3.update(gate_ratio[2], images.size(0))
            gate4.update(gate_ratio[3], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and is_main_task:
                progress.display(i + 1)
            if args.debug:
                break

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    gate1 = AverageMeter('G1', ':6.2f', Summary.AVERAGE)
    gate2 = AverageMeter('G2', ':6.2f', Summary.AVERAGE)
    gate3 = AverageMeter('G3', ':6.2f', Summary.AVERAGE)
    gate4 = AverageMeter('G4', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, gate1, gate2, gate3, gate4],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()
        gate1.all_reduce()
        gate2.all_reduce()
        gate3.all_reduce()
        gate4.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    if is_main_task:
        progress.display_summary()

    return top1.avg, top5.avg, (gate1.avg, gate2.avg, gate3.avg, gate4.avg)

def eval_transfer(val_loader, model_a, model_b, args, is_main_task):
    if args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    atk_a = torchattacks.PGD(
        model_a,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    atk_a.set_normalization_used(mean=mean, std=std)

    atk_b = torchattacks.PGD(
        model_b,
        eps=args.pgd_eps,
        alpha=args.pgd_alpha,
        steps=1 if args.debug else args.pgd_itr,
        random_start=True)
    atk_b.set_normalization_used(mean=mean, std=std)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(model_a):
            delta_a = atk_a(images, target) - images

        with ctx_noparamgrad_and_eval(model_b):
            delta_b = atk_b(images, target) - images

        # compute output
        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)
            p_adv_a = model_a(images+delta_a)
            p_adv_b = model_b(images+delta_b)
            qualified = return_qualified(p_a, p_b, p_adv_a, p_adv_b, target)

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_b2a = model_a((images+delta_b)[qualified, ::])
        p_a2b = model_b((images+delta_a)[qualified, ::])

        acc1_b2a, acc5_b2a = accuracy(p_b2a, target[qualified], topk=(1, 5))
        acc1_a2b, acc5_a2b = accuracy(p_a2b, target[qualified], topk=(1, 5))

        top1_b2a.update(acc1_b2a[0], num_qualified)
        top5_b2a.update(acc5_b2a[0], num_qualified)
        top1_a2b.update(acc1_a2b[0], num_qualified)
        top5_a2b.update(acc5_a2b[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1_b2a = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_a2b = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5_b2a = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top5_a2b = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1_a2b, top1_b2a, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        # if args.distributed:
            # total_qualified.all_reduce()

        # if total_qualified.sum > (num_eval/args.ngpus_per_node):
            # break
        if args.debug:
            break

    if args.distributed:
        top1_b2a.all_reduce()
        top1_a2b.all_reduce()
        top5_b2a.all_reduce()
        top5_a2b.all_reduce()
        total_qualified.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1_a2b.avg, top1_b2a.avg

def eval_transfer_orthogonal(val_loader, model_a, model_b, args, atk_method, is_main_task):
    if args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if args.debug:
        steps = 1
        eps = 4/255
        alpha = 1/255
        num_eval = 100
    else:
        if atk_method.endswith('strong'):
            steps = 40
            eps = 8/255
            alpha = 2/255
        else:
            steps = 20
            eps = 4/255
            alpha = 1/255
        num_eval = 1000

    if atk_method.startswith('linbp'):
        param = {'ord': np.inf,
                 'epsilon': eps,
                 'alpha': alpha,
                 'num_iter': steps,
                 'restarts': 1,
                 'rand_init': True,
                 'clip': True,
                 'loss_fn': nn.CrossEntropyLoss(),
                 'dataset': 'imagenet'}
        attacker = pgd_linbp(**param)
    else:
        if atk_method.startswith('pgd'):

            atk_a = torchattacks.PGD(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps,
                random_start=True)

            atk_b = torchattacks.PGD(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps,
                random_start=True)
        elif atk_method.startswith('mi'):
            atk_a = torchattacks.MIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.MIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('ni'):
            atk_a = torchattacks.NIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.NIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('vni'):
            atk_a = torchattacks.VNIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.VNIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('vmi'):
            atk_a = torchattacks.VMIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.VMIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('sini'):
            atk_a = torchattacks.SINIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.SINIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('ti'):
            atk_a = torchattacks.TIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.TIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)
        elif atk_method.startswith('di'):
            atk_a = torchattacks.DIFGSM(
                model_a,
                eps=eps,
                alpha=alpha,
                steps=steps)

            atk_b = torchattacks.DIFGSM(
                model_b,
                eps=eps,
                alpha=alpha,
                steps=steps)

        atk_a.set_normalization_used(mean=mean, std=std)
        atk_b.set_normalization_used(mean=mean, std=std)

    def run_validate_one_iteration(images, target):
        end = time.time()
        if args.gpu is not None and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.backends.mps.is_available():
            images = images.to('mps')
            target = target.to('mps')
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        with ctx_noparamgrad_and_eval(model_a):
            if atk_method.startswith('linbp'):
                delta_a = attacker.generate(model_a, images, target)
            else:
                delta_a = atk_a(images, target) - images

        with ctx_noparamgrad_and_eval(model_b):
            if atk_method.startswith('linbp'):
                delta_b = attacker.generate(model_b, images, target)
            else:
                delta_b = atk_b(images, target) - images

        # compute output
        with torch.no_grad():
            p_a = model_a(images)
            p_b = model_b(images)
            p_adv_a = model_a(images+delta_a)
            p_adv_b = model_b(images+delta_b)
            qualified = return_qualified(p_a, p_b, p_adv_a, p_adv_b, target)

        # measure accuracy and record loss
        num_qualified = qualified.sum().item()
        p_b2a = model_a((images+delta_b)[qualified, ::])
        p_a2b = model_b((images+delta_a)[qualified, ::])

        acc1_b2a, acc5_b2a = accuracy(p_b2a, target[qualified], topk=(1, 5))
        acc1_a2b, acc5_a2b = accuracy(p_a2b, target[qualified], topk=(1, 5))

        top1_b2a.update(acc1_b2a[0], num_qualified)
        top5_b2a.update(acc5_b2a[0], num_qualified)
        top1_a2b.update(acc1_a2b[0], num_qualified)
        top5_a2b.update(acc5_a2b[0], num_qualified)
        total_qualified.update(num_qualified)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1_b2a = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_a2b = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5_b2a = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top5_a2b = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    total_qualified = AverageMeter('Qualified', ':6.2f', Summary.SUM)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, top1_a2b, top1_b2a, total_qualified],
        prefix='Transfer: ')

    # switch to evaluate mode
    model_a.eval()
    model_b.eval()

    for i, (images, target) in enumerate(val_loader):
        run_validate_one_iteration(images, target)

        if (i % args.print_freq == 0 and is_main_task) or args.debug:
            progress.display(i + 1)

        if args.distributed:
            total_qualified.all_reduce()

        if total_qualified.sum > (num_eval/args.ngpus_per_node):
            break

    if args.distributed:
        top1_b2a.all_reduce()
        top1_a2b.all_reduce()
        top5_b2a.all_reduce()
        top5_a2b.all_reduce()

    if is_main_task:
        progress.display_summary()

    return top1_b2a.avg

