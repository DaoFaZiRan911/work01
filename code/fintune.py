# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/4/7 12:52
# @File    : regression_AVEC2014.py
# @Software: PyCharm
import math
import os
import argparse
import logging
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from my_dataset import MyDataSet
from LIT_Highlight import LitHighlight
from fintune_utils import *
from torch.utils.data import DistributedSampler, SequentialSampler
import torch.distributed as dist
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    device = torch.device(args.device)
    init_distributed_mode(args)
    cudnn.benchmark = True
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_tasks = get_world_size()
    global_rank = get_rank()
    
    if os.path.exists("./AVEC14_path_11") is False:
        os.makedirs("./AVEC14_path_11")
    tb_writer = SummaryWriter()
    
    train_images_path, train_images_label = read_split_data(args.data_path)
    val_images_path, val_images_label = read_split_data(args.test_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    
    sampler_train = DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
    sampler_val = DistributedSampler(
            val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    batch_size = args.batch_size
    if os.path.exists("./AVEC14_logs_11") is False:
        os.makedirs("./AVEC14_logs_11")
    logger = get_logger('./AVEC14_logs_11/finetune.log')
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = DataLoader(train_dataset,
                              sampler=sampler_train,
                              batch_size=batch_size,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            sampler=sampler_val,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn,
                            drop_last=False)

    model = LitHighlight(num_classes=1).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    print("The model has been on the cuda !!!")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        # weights_dict = torch.load(args.weights, map_location=device)["model"]
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict['model'][k]
            
        model.load_state_dict(weights_dict, strict=False)


    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "patch_embed" in name:
                para.requires_grad_(False)
            elif "layers.0" in name:
                para.requires_grad_(False) 
            elif "layers.1" in name:
                para.requires_grad_(False)
            elif "layers.2" in name:
                para.requires_grad_(False)
            elif "layers.3" in name:
                para.requires_grad_(False)
            elif "norm" in name:
                para.requires_grad_(False)
            else:
                para.requires_grad_(True)
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # T_max = 100
    optimizer = optim.AdamW(pg, lr=args.lr, eps=1e-8, weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-8, last_epoch=- 1, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 150, 180], gamma=0.8)
    best_loss = 100.0
    for epoch in range(args.epochs):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)

        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch)

        logger.info('Epoch:[{}]\t '
                    'train_loss={:.4f}\t '
                    'validation_loss={:.4f}'.format(epoch + 1, train_loss, val_loss))
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), "./RMSE_mobile/model-best.pth".format(best_loss))
        torch.save(model.state_dict(), "./AVEC14_path_11/model-{}.pth".format(epoch+1))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', default=0, type=int)

    # TODO 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="./2014train")
    parser.add_argument('--test-path', type=str, default="./2014test")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./path_11/checkpoint-best.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda', help='device')
    
    # DDP
    parser.add_argument('--world_size', default=2, type=int,
                       help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                       help='url used to set up distributed training')

    opt = parser.parse_args()

    main(opt)
