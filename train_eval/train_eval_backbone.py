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
            
            