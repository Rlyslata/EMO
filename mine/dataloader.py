import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据转换和预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 数据集路径
data_dir = '/home/dl/data/chest-xray-pneumonia/train'

# 加载数据集
train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 打印类映射
print("Class to index mapping:", train_dataset.class_to_idx)

# 获取一个批次的数据和标签
data_iter = iter(train_loader)
images, labels = next(data_iter)

# 打印标签和其对应的类别名称
print("Labels:", labels)
print("Label indices:", labels.numpy())
print("Label names:", [train_dataset.classes[label] for label in labels])
