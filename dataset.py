import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from utils import compute_reprojection  #

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'image_left')) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, 'image_left', img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        flow_name = f"{int(img_name.split('_')[0]):06d}_{int(img_name.split('_')[0])+1:06d}_flow.npy"
        flow_path = os.path.join(self.root_dir, 'flow', flow_name)
        if not os.path.exists(flow_path):
            print(f"Flow file not found: {flow_path}, skipping this sample.")
            return self.__getitem__((idx + 1) % len(self))
        flow = np.load(flow_path)

        return image, torch.from_numpy(flow)

class ReprojectionDataset(Dataset):
    def __init__(self, root_dir, transformation_matrix):
        self.root_dir = root_dir
        self.transformation_matrix = transformation_matrix
        reprojection_data_path = os.path.join(root_dir, 'reprojection_data.npy')
        if not os.path.exists(reprojection_data_path):
            print(f"Reprojection data file not found: {reprojection_data_path}, creating a new one.")
            # 假设我们有一个原始点数据文件 'original_points.npy'
            original_points_path = os.path.join(root_dir, 'original_points.npy')
            if not os.path.exists(original_points_path):
                raise FileNotFoundError(f"Original points data file not found: {original_points_path}")
            original_points = np.load(original_points_path)
            reprojected_points = compute_reprojection(original_points, transformation_matrix)
            # 假设 delta_x, delta_y, weight 是一些计算结果，这里用随机数代替
            delta_x = np.random.rand(reprojected_points.shape[0], 1)
            delta_y = np.random.rand(reprojected_points.shape[0], 1)
            weight = np.random.rand(reprojected_points.shape[0], 1)
            example_data = np.hstack([reprojected_points, delta_x, delta_y, weight])
            np.save(reprojection_data_path, example_data)
        self.data = np.load(reprojection_data_path)
        self.reprojected_data = self.compute_reprojection(self.data[:, :3])

    def compute_reprojection(self, points):
        return compute_reprojection(points, self.transformation_matrix)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point = self.reprojected_data[idx]  # 重投影后的点
        target = self.data[idx, 3:]  # delta_x, delta_y, weight
        return torch.from_numpy(point).float(), torch.from_numpy(target).float()