import json
import math
import os
import random
import sys
from LIT_Highlight import LitHighlight
from PIL import Image
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


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


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
# todo
data_path = './2014test'
#if os.path.exists("./RMSE_gai") is False:
 #   os.makedirs("./RMSE_gai")
img_size = 224
batch_size = 1
weights = './AVEC14_path_11/model-30.pth'
torch.cuda.manual_seed_all(0)
images_path, label = read_split_data(data_path)
data_transform = transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                     transforms.CenterCrop(img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = MyDataSet(images_path=images_path,
                    images_class=label,
                    transform=data_transform)
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])
loader = torch.utils.data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=nw,
                                     drop_last=False,
                                     collate_fn=dataset.collate_fn)
val_num = len(loader.dataset)
print(val_num)
# model = create_model(num_classes=1).to(device)
model = LitHighlight(num_classes=1).to(device)
model = torch.nn.DataParallel(model, [0, 1])
# model = LKCT(1).to(device)
print("The model has been on the cuda !!!")
if weights != "":
    assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
    # weights_dict = torch.load(args.weights, map_location=device)["model"]
    weights_dict = torch.load(weights, map_location=device)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    with torch.no_grad():
        loss_function = torch.nn.MSELoss()
        loss_mae = torch.nn.L1Loss()
        
        accu_loss = torch.zeros(1).to(device)
        accu_mae = torch.zeros(1).to(device)
        data_loader = tqdm(loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, targets = data
            labels = targets.unsqueeze(1).to(torch.float32)
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))
            mae_loss = loss_mae(pred, labels.to(device))
            accu_mae += mae_loss
            accu_loss += loss
    test_loss = math.sqrt(accu_loss.item() / (step + 1))
    test_mae = accu_mae.item() / (step + 1)
    print('The RMSE loss on the test dataset is : ', test_loss)
    print('The MAE loss on the test dataset is :', test_mae)
    

