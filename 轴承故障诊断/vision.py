import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import pandas as pd

# 加载你的自定义数据集
train_df = pd.read_csv(r'C:\Users\张\Desktop\Bearing fault detection\train_set.csv')
test_df = pd.read_csv(r'C:\Users\张\Desktop\Bearing fault detection\test_set.csv')

train_imgs = train_df['image_path'].values
train_labels = train_df['label'].values
test_imgs = test_df['image_path'].values
test_labels = test_df['label'].values

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((96, 96)),
    torchvision.transforms.ToTensor()
])

class Mydataset(torch.utils.data.Dataset):
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
            nn.Linear(64 * 12 * 12, 64),  # 修改维度以适应新的图像大小
            nn.ReLU(),
            nn.Linear(64, 4)  # 修改为4分类
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 加载模型并评估
cnnnet = CNNnet().to(device)
cnnnet.load_state_dict(torch.load("best_model.pth"))
cnnnet.eval()

features = []
labels = []
with torch.no_grad():
    for x_test, y_test in test_dataloader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = cnnnet(x_test)
        features.append(outputs)
        labels.append(y_test)

# 将特征和标签转换为 CPU 张量
features = torch.cat(features, dim=0).cpu().numpy()
labels = torch.cat(labels, dim=0).cpu().numpy()

# t-SNE 可视化（带 PCA 降维）
# 计算特征的实际维度
n_samples, n_features = features.shape
n_components = min(50, n_features, n_samples - 1)  # 保证 n_components 不超过 n_samples 或 n_features

# PCA 预降维
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features)

# 使用 t-SNE 降维至 2 维
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
tsne_results = tsne.fit_transform(features_pca)

# 处理缺失的类别
unique_labels = np.unique(labels)
if len(unique_labels) != 4:
    missing_labels = set(range(4)) - set(unique_labels)
    for label in missing_labels:
        tsne_results = np.vstack([tsne_results, [np.nan, np.nan]])
        labels = np.append(labels, label)

# 可视化结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE Visualization after Model Training')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['B', 'IR', 'NO', 'OR'])
plt.show()

# 测试集计算各种指标
y_pred = np.argmax(features, axis=1)
acc = accuracy_score(labels, y_pred)
pre = precision_score(labels, y_pred, average='macro', zero_division=0)
recall = recall_score(labels, y_pred, average='macro', zero_division=0)
f1score = f1_score(labels, y_pred, average='macro', zero_division=0)

print('计算指标结果：\nAcc: %.2f%% \nPre: %.2f%% \nRecall: %.2f%% \nF1-score: %.2f%% ' % (100 * acc, 100 * pre, 100 * recall, 100 * f1score))

# 绘制混淆矩阵
cm = confusion_matrix(labels, y_pred, labels=[0, 1, 2, 3])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'IR', 'NO', 'OR'])
disp.plot()
plt.show()
