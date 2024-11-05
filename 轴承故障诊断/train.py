import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader


# 自定义数据集类
class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# 加载训练和测试数据
train_df = pd.read_csv(r'C:\Users\张\Desktop\Bearing fault detection\train_set.csv')
test_df = pd.read_csv(r'C:\Users\张\Desktop\Bearing fault detection\test_set.csv')

train_imgs = train_df['image_path'].values
train_labels = train_df['label'].values
test_imgs = test_df['image_path'].values
test_labels = test_df['label'].values

train_dataset = Mydataset(train_imgs, train_labels, transform)
test_dataset = Mydataset(test_imgs, test_labels, transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 构造CNN网络
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 修改为4分类
        )

    def forward(self, x):
        x = self.model(x)
        return x


cnnnet = CNNnet().to(device)
optimizer = torch.optim.Adam(cnnnet.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss().to(device)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 20
best_acc = 0.0

# 训练和验证循环
for i in range(epoch):
    print(f'第{i + 1}轮训练开始')

    # 训练步骤
    cnnnet.train()
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = cnnnet(imgs)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数 {total_train_step}，loss {loss.item()}")

    # 验证步骤
    cnnnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = cnnnet(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / len(test_dataset)
    print(f'整体测试集上的loss：{total_test_loss}')
    print(f'整体测试集上的正确率：{100 * avg_accuracy:.2f}%')

    total_test_step += 1
    if avg_accuracy > best_acc:
        best_acc = avg_accuracy
        print(f'目前最好的正确率：{best_acc}')
        torch.save(cnnnet.state_dict(), "best_model.pth")
        print(f'Epoch{i + 1}: 模型已保存, 验证准确率：{100 * avg_accuracy:.2f}%')
