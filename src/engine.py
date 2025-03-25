import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from tqdm import tqdm
import json
import os

def pretrain_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast,
                    log_writer=None,
                    model_without_ddp=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    steps_of_one_epoch = len(data_loader)

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step
        metric_logger.update(steps=int(steps))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        samples, _ = data
        samples = samples.to(device, non_blocking=True)

        with amp_autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='step'+str(steps))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples, targets = data
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        pred_all.extend(pred[0].cpu())
        target_all.extend(targets.cpu())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        if isinstance(loss_scaler, NativeScaler):
            loss /= accum_iter
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=False,
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, if_stat=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    pred_all = []
    target_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item()/100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item()/100, n=batch_size)

    macro = precision_recall_fscore_support(target_all, pred_all, average='weighted')
    cm = confusion_matrix(target_all, pred_all)

    # compute acc, precision, recall, f1 for each class
    acc = accuracy_score(target_all, pred_all)
    pre_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(target_all, pred_all, average=None)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(
        '* Pre {macro_pre:.4f} Rec {macro_rec:.4f} F1 {macro_f1:.4f}'
        .format(macro_pre=macro[0], macro_rec=macro[1],
                    macro_f1=macro[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['weighted_pre'] = macro[0].tolist()
    test_state['weighted_rec'] = macro[1].tolist()
    test_state['weighted_f1'] = macro[2].tolist()
    test_state['cm'] = cm.tolist()
    test_state['acc'] = acc.tolist()
    test_state['pre_per_class'] = pre_per_class
    test_state['rec_per_class'] = rec_per_class
    test_state['f1_per_class'] = f1_per_class
    test_state['support_per_class'] = support_per_class

    return test_state


import time
import gc

@torch.no_grad()
def evaluate_speed_test(data_loader, model, device, args):
    # switch to evaluation mode
    model.eval()
    model_mem = torch.cuda.memory_allocated() / (1024**2)
    res_list = []
    for i in tqdm(range(3, 11), desc="Batch size"):
        # reset memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()
        batch_size = 2 ** i
        if i == 3:
            batch_size = 2 ** 10
        data_loader_tmp = torch.utils.data.DataLoader(data_loader.dataset, sampler=data_loader.sampler,
                            batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)
        pred_all = []
        start_time = time.time()
        for batch in data_loader_tmp:
            if time.time() - start_time > 30:
                break
            images = batch[0]
            # target = batch[-1]
            images = images.to(device, non_blocking=True)
            # target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                # loss = criterion(output, target)

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            pred_all.extend(pred[0].cpu())
            # target_all.extend(target.cpu())
        end_time = time.time()
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)
        torch.cuda.empty_cache()
        gc.collect()

        if i == 3: # only for warming up
            continue
        res_list.append({
            "batch size": batch_size,
            "time (s)": end_time - start_time,
            "total smaples": len(pred_all),
            "speed (sample per second)": len(pred_all) / (end_time - start_time),
            "max memory consumption (MB)": max_mem,
            "model memory consumption (MB)": model_mem
        })
   
    with open(os.path.join(args.output_dir, "speed_test.json"), "w") as f:
        json.dump(res_list, f, indent=2)

    return None