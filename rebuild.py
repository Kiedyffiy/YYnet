import torch
import random
import trimesh
import os
from torch.utils.data import DataLoader
from mesh_to_pc import (
    rebuild_3d_model,
)
from dataread import GLBDataset


# 假设已经定义好了你的 dataloader, rebuild_3d_model 函数

# 定义保存 3D 模型的函数
def save_3d_model_as_obj(mesh, path):
    """
    将 Trimesh 对象保存为 .obj 文件
    :param mesh: Trimesh 对象
    :param path: 保存路径
    """
    mesh.export(path)
    #print(f"3D 模型已保存至: {path}")

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

def export_models(face_coords,mask,save_dir):
    current_device = torch.cuda.current_device()
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
            model_save_path = os.path.join(save_dir, f"cuda_{current_device}_downsample_{downsample_idx+1}_sample_{sample_idx+1}.obj")

            # 保存 3D 模型
            save_3d_model_as_obj(mesh, model_save_path)

def export_cpmodels(face_coords, target_triangles, mask, save_dir, spacing=10):
    current_device = torch.cuda.current_device()

    # 初始化用于存储所有模型顶点和面的列表
    all_vertices = []
    all_faces = []
    vertex_offset = 0  # 用于记录顶点索引偏移量

    # 假设降采样维度为第二维度（num of downsampletimes），这里我们处理每个降采样版本
    for downsample_idx in range(face_coords.shape[1]):  # 处理每个降采样版本
        # 提取对应的 face_coord 和 mask
        downsampled_face_coord = face_coords[:, downsample_idx, :, :, :]  # 第 downsample_idx 层降采样的 face_coord
        downsampled_target_coord = target_triangles[:, downsample_idx, :, :, :]  # 第 downsample_idx 层 target_triangle
        downsampled_mask = mask[:, downsample_idx, :]  # 对应的 mask

        # 对每个样本单独处理
        for sample_idx in range(downsampled_face_coord.shape[0]):
            face_coord_sample = downsampled_face_coord[sample_idx]
            target_coord_sample = downsampled_target_coord[sample_idx]
            mask_sample = downsampled_mask[sample_idx]

            # 调用重建函数重建预测的 3D 模型
            pred_mesh = rebuild_3d_model(face_coord_sample, mask_sample)
            target_mesh = rebuild_3d_model(target_coord_sample, mask_sample)

            # 将预测模型沿 x 轴位移排列
            pred_vertices = pred_mesh.vertices + [spacing * sample_idx, 0, 0]
            pred_faces = pred_mesh.faces + vertex_offset

            # 将目标模型沿 x 轴位移，保持在预测模型旁边
            target_vertices = target_mesh.vertices + [spacing * sample_idx + spacing / 2, 0, 0]
            target_faces = target_mesh.faces + vertex_offset + len(pred_vertices)

            # 添加预测模型和目标模型的顶点和面信息
            all_vertices.extend(pred_vertices)
            all_faces.extend(pred_faces)

            all_vertices.extend(target_vertices)
            all_faces.extend(target_faces)

            # 更新顶点偏移量
            vertex_offset += len(pred_vertices) + len(target_vertices)

    # 创建最终的 trimesh 对象
    combined_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

    # 设置保存路径
    model_save_path = os.path.join(save_dir, f"cuda_{current_device}_combined.obj")

    # 保存 3D 模型
    save_3d_model_as_obj(combined_mesh, model_save_path)


# 假设 dataloader 已经定义好
# 运行导出 3D 模型的函数
#export_random_models(dataloader, num_models=5, save_dir=savepath)