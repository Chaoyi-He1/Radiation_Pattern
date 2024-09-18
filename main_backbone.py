import argparse
import datetime
import glob
import json
import math
import os
import random
import time
import pickle
from pathlib import Path
import torch
import torch.utils.data
import tempfile
from tensorboardX import SummaryWriter

from models.backbone import Pointnet
from datasets.dataset import Pointnet_Dataset


def get_parser():
    parser = argparse.ArgumentParser()
    
    # dataset parameters
    parser.add_argument('--data_path', default='data', type=str, help='path to dataset')

    # model parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_points', default=1024, type=int, help='number of points')
    parser.add_argument('--num_classes', default=40, type=int, help='number of classes')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lrf', default=0.1, type=float, help='learning rate final')

    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--log_interval', default=10, type=int, help='log interval')
    parser.add_argument('--save_interval', default=10, type=int, help='save interval')
    parser.add_argument('--resume', default='weight/model_100.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model')
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--log_dir', default='logs', type=str, help='directory to save logs')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    return parser


def main(args):
    
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    
    device = torch.device(args.device)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # create model
    model = Pointnet(num_points=args.num_points, num_classes=args.num_classes)
    
    # load pretrained model
    if args.resume.endswith('.pth'):
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(device)
    
    # create data loader
    whole_dataset = Pointnet_Dataset(data_folder_path=args.data_path, is_train=True, num_points=args.num_points)
    train_data, val_data = torch.utils.data.random_split(whole_dataset, [int(0.8*len(whole_dataset)), len(whole_dataset) - int(0.8*len(whole_dataset))])
    
    train_sampler = torch.utils.data.RandomSampler(train_data)
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, sampler=val_sampler)
    
    # create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = 0
    
    # start training
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}/{args.epochs}')
        
        # train for one epoch
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, epoch, tb_writer)
        
        # update learning rate
        scheduler.step()
        
        # evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, tb_writer)
        
        # log results in tensorboard
        tb_writer.add_scalar('train/loss', train_loss, epoch)
        tb_writer.add_scalar('train/accuracy', train_acc, epoch)
            
        tb_writer.add_scalar('val/loss', val_loss, epoch)
        tb_writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # save checkpoint
        if epoch % 10 == 0:
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(save_dict, os.path.join(args.save_dir, f'model_{epoch}.pth'))
    return


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)