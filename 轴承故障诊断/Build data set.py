import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd

# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
    def __init__(self, root):
        self.imgs_path = root

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path)
        img = np.array(img)
        return img, img_path

    def __len__(self):
        return len(self.imgs_path)

# 使用glob方法来获取数据图片的所有路径
base_dir = r'C:\Users\张\Desktop\Bearing fault detection\CTGU_GADF_64'
categories = ['B', 'IR', 'NO', 'OR']
train_imgs_path = []
test_imgs_path = []
train_labels = []
test_labels = []
category_counts = {}

for category in categories:
    category_imgs_path = glob.glob(f'{base_dir}\\{category}\\*.png')
    np.random.shuffle(category_imgs_path)
    train_size = int(len(category_imgs_path) * 0.8)
    train_imgs_path += category_imgs_path[:train_size]
    train_labels += [category] * train_size
    test_imgs_path += category_imgs_path[train_size:]
    test_labels += [category] * (len(category_imgs_path) - train_size)
    category_counts[category] = len(category_imgs_path)

# 打印每个类别的图片数量，确保所有类别的图片路径正确加载
for category, count in category_counts.items():
    print(f'类别 {category} 图片数量: {count}')

# 检查训练集和测试集的标签和路径长度是否匹配
print(f'训练集图片数量: {len(train_imgs_path)}, 标签数量: {len(train_labels)}')
print(f'测试集图片数量: {len(test_imgs_path)}, 标签数量: {len(test_labels)}')

# 转换标签为数值形式
species_to_id = {'B': 0, 'IR': 1, 'NO': 2, 'OR': 3}
train_labels = [species_to_id[label] for label in train_labels]
test_labels = [species_to_id[label] for label in test_labels]

# 创建DataFrame并保存为CSV
train_df = pd.DataFrame({'image_path': train_imgs_path, 'label': train_labels})
test_df = pd.DataFrame({'image_path': test_imgs_path, 'label': test_labels})

# 保存为CSV文件
train_df.to_csv(r'C:\Users\张\Desktop\Bearing fault detection\train_set.csv', index=False)
test_df.to_csv(r'C:\Users\张\Desktop\Bearing fault detection\test_set.csv', index=False)

print("训练集和测试集已保存为CSV文件")
