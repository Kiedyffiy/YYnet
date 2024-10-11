import torch
import random
import trimesh
import os
from torch.utils.data import DataLoader
from mesh_to_pc import (
    rebuild_3d_model,
)
from dataread import GLBDataset

batch_size = 16

pkl_file_path =  '/root/data/YYnetdata/20241009_010626/dataset.pkl'

dataset = GLBDataset.load_dataset(pkl_file_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

savepath = "/root/data/YYnetrebuild/v1"
# 假设已经定义好了你的 dataloader, rebuild_3d_model 函数

# 定义保存 3D 模型的函数
def save_3d_model_as_obj(mesh, path):
    """
    将 Trimesh 对象保存为 .obj 文件
    :param mesh: Trimesh 对象
    :param path: 保存路径
    """
    mesh.export(path)
    print(f"3D 模型已保存至: {path}")

# 从 dataloader 中随机挑选 5 个样本
def export_random_models(dataloader, num_models=5, save_dir="./3d_models"):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 随机挑选 5 个样本
    all_batches = list(dataloader)
    selected_batches = random.sample(all_batches, num_models)

    for i, batch in enumerate(selected_batches):
        point_feature, face_coords, mask = batch

        # 假设降采样维度为第二维度（num of downsampletimes），这里我们处理每个降采样版本
        for downsample_idx in range(face_coords.shape[1]):  # 处理每个降采样版本
            # 提取对应的 face_coord 和 mask
            downsampled_face_coord = face_coords[:, downsample_idx, :, :, :]  # 提取第 downsample_idx 层降采样的 face_coord
            downsampled_mask = mask[:, downsample_idx, :]  # 提取对应的 mask

            # 对每个样本单独重建
            for sample_idx in range(downsampled_face_coord.shape[0]):
                face_coord_sample = downsampled_face_coord[sample_idx]
                mask_sample = downsampled_mask[sample_idx]

                # 调用重建函数
                mesh = rebuild_3d_model(face_coord_sample, mask_sample)

                # 设置保存路径
                model_save_path = os.path.join(save_dir, f"model_{i+1}_downsample_{downsample_idx+1}_sample_{sample_idx+1}.obj")

                # 保存 3D 模型
                save_3d_model_as_obj(mesh, model_save_path)

# 假设 dataloader 已经定义好
# 运行导出 3D 模型的函数
export_random_models(dataloader, num_models=5, save_dir=savepath)
