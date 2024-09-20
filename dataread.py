import os
import torch
from torch.utils.data import Dataset, DataLoader
from mesh_to_pc import process_shapenet_models
import pickle
from datetime import datetime

class GLBDataset(Dataset):
    def __init__(self, datapath, k = 800 , marching_cubes=True, sample_num=4096, save_path=None):
        # 过滤掉面数超过k的模型并生成数据集
        self.pc_normal_list, self.return_mesh_list, self.face_coods, self.mask = process_shapenet_models(
            datapath, k=k, marching_cubes=marching_cubes, sample_num=sample_num
        )
        
        # 可选：保存处理后的数据集
        if save_path is not None:
            self.save_dataset(save_path)

    def __len__(self):
        return len(self.pc_normal_list)

    def __getitem__(self, idx):
        pc_normal = self.pc_normal_list[idx]
        mesh = self.return_mesh_list[idx].vertices
        face_coord = self.face_coods[idx]
        mask = self.mask[idx]

        return pc_normal, mesh, face_coord, mask

    def save_dataset(self, save_path):
    # 获取当前时间并格式化为字符串
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建以时间为名称的文件夹
        save_dir = os.path.join(save_path, timestamp)
        os.makedirs(save_dir, exist_ok=True)
    
    # 生成保存文件的路径
        file_path = os.path.join(save_dir, 'dataset.pkl')
    
    # 将数据集保存为pickle文件
        with open(file_path, 'wb') as f:
            pickle.dump({
                'pc_normal_list': self.pc_normal_list,
                'return_mesh_list': self.return_mesh_list,
                'face_coods': self.face_coods,
                'mask': self.mask
            }, f)
    
        print(f"Dataset saved to {file_path}")

    @staticmethod
    def load_dataset(load_path):
        # 从pickle文件加载数据集
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        dataset = GLBDataset.__new__(GLBDataset)  # 使用__new__方法创建实例
        dataset.pc_normal_list = data['pc_normal_list']
        dataset.return_mesh_list = data['return_mesh_list']
        dataset.face_coods = data['face_coods']
        dataset.mask = data['mask']
        return dataset