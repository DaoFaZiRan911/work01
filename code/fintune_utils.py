# -*- coding: utf-8 -*-
# @Author  : Lan Zhang
# @Time    : 2022/7/12 14:16
# @File    : utils.py
# @Software: PyCharm
import os
import sys
import json
import pickle
import math
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist


# def read_split_data(root: str, val_rate: float = 0.2):
#     random.seed(0)  # 保证随机结果可复现
#     assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

#     # 遍历文件夹，一个文件夹对应一个类别
#     people_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
#     # 排序，保证顺序一致
#     people_class.sort(key=lambda x: int(x))
#     # 生成类别名称以及对应的数字索引
#     class_indices = dict((k, v) for v, k in enumerate(people_class))
#     json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)

#     train_images_path = []  # 存储训练集的所有图片路径
#     train_images_label = []  # 存储训练集图片对应索引信息
#     val_images_path = []  # 存储验证集的所有图片路径
#     val_images_label = []  # 存储验证集图片对应索引信息
#     every_class_num = []  # 存储每个类别的样本总数
#     supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
#     # 遍历每个文件夹下的文件
#     for cla in people_class:
#         cla_path = os.path.join(root, cla)
#         # 遍历获取supported支持的所有文件路径
#         images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
#                   if os.path.splitext(i)[-1] in supported]
#         # 获取该类别对应的索引
#         image_class = class_indices[cla]
#         # 记录该类别的样本数量
#         every_class_num.append(len(images))
#         # 按比例随机采样验证样本
#         val_path = random.sample(images, k=int(len(images) * val_rate))

#         for img_path in images:
#             if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
#                 val_images_path.append(img_path)
#                 val_images_label.append(image_class)
#             else:  # 否则存入训练集
#                 train_images_path.append(img_path)
#                 train_images_label.append(image_class)

#     print("{} images were found in the dataset.".format(sum(every_class_num)))
#     print("{} images for training.".format(len(train_images_path)))
#     print("{} images for validation.".format(len(val_images_path)))

#     plot_image = False
#     if plot_image:
#         # 绘制每种类别个数柱状图
#         plt.bar(range(len(people_class)), every_class_num, align='center')
#         # 将横坐标0,1,2,3,4替换为相应的类别名称
#         plt.xticks(range(len(people_class)), people_class)
#         # 在柱状图上添加数值标签
#         for i, v in enumerate(every_class_num):
#             plt.text(x=i, y=v + 5, s=str(v), ha='center')
#         # 设置x坐标
#         plt.xlabel('image class')
#         # 设置y坐标
#         plt.ylabel('number of images')
#         # 设置柱状图的标题
#         plt.title(' class distribution')
#         plt.show()

#     return train_images_path, train_images_label, val_images_path, val_images_label
def read_split_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    people_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    people_class.sort(key=lambda x: int(x))
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(people_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []  # 存储训练集的所有图片路径
    test_images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in people_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(people_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(people_class)), people_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title(' class distribution')
        plt.show()

    return test_images_path, test_images_label

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):

    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = False

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()  # MSE
    # loss_function = torch.nn.L1Loss()  # MAE
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    train_num = len(data_loader.dataset)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        # labels = -1 + (2 / 63) * labels
        labels = labels.unsqueeze(1).to(torch.float32)
        pred = model(images.to(device))
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        accu_loss += loss.detach()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print("############", name)
        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, math.sqrt(accu_loss.item() / (step+1)))
        # accu_num.item() / sample_num)
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        result = math.sqrt(accu_loss.item() / (step+1))
        optimizer.step()

    return result


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.MSELoss()
    loss_mae = torch.nn.L1Loss()
    # loss_function = torch.nn.L1Loss()  # MAE
    model.eval()

    # accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_mae = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    val_num = len(data_loader.dataset)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        # labels = -1 + (2 / 63) * labels
        labels = labels.unsqueeze(1).to(torch.float32)
        pred = model(images.to(device))
        # pred_classes = torch.max(pred, dim=1)[1]
        # accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        mae_loss = loss_mae(pred, labels.to(device))
        accu_mae += mae_loss
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] RMSE loss: {:.3f} MAE loss: {:.3f}".format(epoch, math.sqrt(accu_loss.item() / (step+1)),
                                                                                        accu_mae.item() / (step + 1))
        result = math.sqrt(accu_loss.item() / (step+1))

    return result
