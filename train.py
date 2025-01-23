import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import FeatureMatchingNet, ReprojectionNet
from dataset import ImageDataset, ReprojectionDataset
import numpy as np

def train_feature_matching(root_dir, num_epochs=10, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FeatureMatchingNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for images, flows in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            flows = flows.view(batch_size, -1, 2).mean(dim=1)  # 调整flows的尺寸以匹配outputs
            if flows.size(0) != outputs.size(0):
                continue  # 跳过批量大小不匹配的样本
            loss = criterion(outputs, flows)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'feature_matching_model.pth')

def train_reprojection(root_dir, num_epochs=10, batch_size=32):
    transformation_matrix = np.eye(4)  # 示例变换矩阵，可以根据需要进行修改
    dataset = ReprojectionDataset(root_dir, transformation_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ReprojectionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for points, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'reprojection_model.pth')

if __name__ == "__main__":
    root_dir = "P001"
    train_feature_matching(root_dir)
    train_reprojection(root_dir)