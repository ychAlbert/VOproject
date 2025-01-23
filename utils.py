import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def extract_patches(image, patch_size=8):
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches

def compute_reprojection(points, transformation_matrix):
    """
    计算重投影数据
    :param points: 原始点 (N, 3)
    :param transformation_matrix: 变换矩阵 (4, 4)
    :return: 重投影后的点 (N, 3)
    """
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # 转换为齐次坐标
    reprojected_points = points_homogeneous @ transformation_matrix.T  # 应用变换矩阵
    reprojected_points /= reprojected_points[:, -1:]  # 转换回非齐次坐标
    return reprojected_points[:, :3]