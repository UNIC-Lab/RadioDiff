from PIL import Image
import numpy as np
import os

# 定义计算NMSE的函数
def calculate_nmse(pred, gt):
    return np.sum((pred - gt) ** 2) / np.sum(gt ** 2)

# 定义文件夹路径
test_path = "/mnt/mydisk/kdtao/DiffRadio/results/test"
gt_path = "/mnt/mydisk/kdtao/RadioMapSeer/gain/DPM"

# 获取预测结果文件夹中所有图片的文件名
test_images = os.listdir(test_path)

# 初始化用于存储所有NMSE值的列表
nmse_values = []

# 遍历每个预测图片
for img_name in test_images[100:]:
    # 加载预测图片和对应的GT图片
    pred_img = np.array(Image.open(os.path.join(test_path, img_name)).convert('L'))
    gt_img = np.array(Image.open(os.path.join(gt_path, img_name)))

    # 确保图片尺寸相同
    if pred_img.shape == gt_img.shape:
        # 计算并存储NMSE
        nmse = calculate_nmse(pred_img, gt_img)
        print(nmse)
        nmse_values.append(nmse)

# 计算所有图片对的平均NMSE
average_nmse = np.mean(nmse_values)
print(average_nmse)
