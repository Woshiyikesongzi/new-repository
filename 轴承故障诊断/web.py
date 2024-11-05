import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from torch import nn
import os

# 定义CNN模型
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
            nn.Linear(64*12*12, 64),  # 确保维度正确
            nn.ReLU(),
            nn.Linear(64, 4)  # 修改为4分类
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 确保 best_model.pth 文件在同一目录下
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
model = CNNnet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 定义类别映射字典
class_names = {0: '滚动体故障', 1: '内圈故障', 2: '正常', 3: '外圈故障'}

# 图像预处理
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # 确保图像尺寸与训练时一致
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Streamlit 应用
st.title('Bearing fault diagnosis Web App')
uploaded_file = st.file_uploader('Choose an image...', type='png')
if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("正在预测，请稍等...")
    image = transform_image(image)
    with torch.no_grad():
        output = model(image)
    predicted_class = output.argmax(1).item()
    st.write(f'预测结果: {class_names[predicted_class]}')
