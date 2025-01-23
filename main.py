import torch
from model import FeatureMatchingNet, ReprojectionNet
from utils import preprocess_image, extract_patches

def main():
    # 加载训练好的模型
    feature_matching_model = FeatureMatchingNet()
    feature_matching_model.load_state_dict(torch.load('feature_matching_model.pth', weights_only=True))
    feature_matching_model.eval()

    reprojection_model = ReprojectionNet()
    reprojection_model.load_state_dict(torch.load('reprojection_model.pth', weights_only=True))
    reprojection_model.eval()

    # 特征点匹配和位移计算
    image_path = 'D:\\ZhangLe_paper\\VOproject\P001\\image_right\\000000_right.png'
    image = preprocess_image(image_path)
    patches = extract_patches(image)

    with torch.no_grad():
        displacements = feature_matching_model(patches)

    print("Patch displacements:", displacements)

    # 重投影校正
    reprojection_point = torch.tensor([1.0, 2.0, 3.0]).float().unsqueeze(0)  # 示例重投影点

    with torch.no_grad():
        reprojection_output = reprojection_model(reprojection_point)

    delta_x, delta_y, weight = reprojection_output[0]
    print(f"Reprojection correction: delta_x={delta_x.item():.4f}, delta_y={delta_y.item():.4f}, weight={weight.item():.4f}")

if __name__ == "__main__":
    main()