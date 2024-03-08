import math
import torch
from torch.nn import functional as F
from util.misc import *
import torch.nn as nn
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0.01,
                    scaler=None):
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    
    for input_1, input_2 in metric_logger.log_every(data_loader, 50, header):
        input_1 = input_1.to(device)
        input_2 = input_2.to(device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output_1 = model(input_1)
            output_2 = model(input_2)
            loss = criterion(output_1, output_2)
        
        # reduce losses over all GPUs for logging purposes
        loss_reduced = reduce_loss(loss)
        
        # Backward pass
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        # Update metric logger
        metric_logger.update(loss=loss_reduced.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        