import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
import time

from src.utils_log import Summary, AverageMeter, ProgressMeter
from src.evaluation import accuracy
import ipdb

AVOID_ZERO_DIV = 1e-6

def train(train_loader, model, criterion, optimizer, epoch, device, args, is_main_task, scaler, mixup_cutmix):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # mixup-cutmix
        orig_target = target.clone().detach()
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            images, target = mixup_cutmix(images, target)

        # compute output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, orig_target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_task:
            progress.display(i + 1)
        if args.debug and i == 2:
            break

    return top1.avg, top5.avg, losses.avg

def train_gate(train_loader, model, criterion, optimizer, epoch, device, args, is_main_task, scaler, mixup_cutmix):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # mixup-cutmix
        orig_target = target.clone().detach()
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            images, target = mixup_cutmix(images, target)

        # compute output
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            gates, logits = model(images)
            # concatgate = torch.cat([gates[0], gates[1], gates[2], gates[3], gates[4]], dim=1).squeeze()
            # loss = criterion(output, target)
            # pred1 = F.softmax(logits[:,:,0], dim=1)
            # pred2 = F.softmax(logits[:,:,1], dim=1)
            # pred3 = F.softmax(logits[:,:,2], dim=1)
            # pred4 = F.softmax(logits[:,:,3], dim=1)
            # pred5 = F.softmax(logits[:,:,4], dim=1)

            # onehotlabels = torch.zeros_like(pred1)
            # onehotlabels[range(len(target)), target] = 1

            pred = F.softmax(logits, dim=1)
            onehotlabels_new = torch.zeros_like(pred)
            onehotlabels_new[range(len(target)), target, :] = 1
            decision_from_gate = (-onehotlabels_new*pred).sum(dim=1).argmin(dim=1)

            # loss1 = torch.sum(-onehotlabels*pred1, dim=1)
            # loss2 = torch.sum(-onehotlabels*pred2, dim=1)
            # loss3 = torch.sum(-onehotlabels*pred3, dim=1)
            # loss4 = torch.sum(-onehotlabels*pred4, dim=1)
            # loss5 = torch.sum(-onehotlabels*pred5, dim=1)

            # concatloss = torch.cat((loss1.unsqueeze(1),
                                    # loss2.unsqueeze(1),
                                    # loss3.unsqueeze(1),
                                    # loss4.unsqueeze(1),
                                    # loss5.unsqueeze(1)), dim=1)

            # llabels = torch.argmin(concatloss, dim=1)

            loss = criterion(gates, decision_from_gate)
            for idx in range(logits.shape[2]):
                loss += criterion(logits[:, :, idx], target)
            loss = loss.mean()

        maxidx = torch.max(gates, dim=1)[1]
        acc1 = (pred[range(len(maxidx)), :, maxidx].argmax(dim=1) == target).sum()/images.size(0)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, orig_target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and is_main_task:
            progress.display(i + 1)
        if args.debug and i == 2:
            break

    return top1.avg, top5.avg, losses.avg
