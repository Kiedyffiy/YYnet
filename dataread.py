import os
import torch
from torch.utils.data import Dataset, DataLoader
from mesh_to_pc import process_shapenet_models
import pickle
from datetime import datetime
#import open3d as o3d
import trimesh
import numpy as np

class GLBDataset(Dataset):
    def __init__(self, datapath, category_list, k = 800 , marching_cubes=True, sample_num=4096, save_path=None):
        # 过滤掉面数超过k的模型并生成数据集
        self.pc_normal_list, self.return_mesh_list, self.face_coods, self.mask = process_shapenet_models(
            datapath, category_list, k=k, marching_cubes=marching_cubes, sample_num=sample_num
        )
        self.downsampletimes = self.mask.shape[1]
        # 可选：保存处理后的数据集
        if save_path is not None:
            self.save_dataset(save_path)

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        #pc_normal = self.pc_normal_list[idx]
        #mesh = self.return_mesh_list[idx] #减小内存占用
        #print(mesh.shape)
        #pointfeature = self.point_feature[idx]
        pc_normal_sublist = self.pc_normal_list[idx]
        return_mesh_sublist = self.return_mesh_list[idx]
        normalized_pc_normal_list = []
        for i in range(self.downsampletimes):
            vertices = return_mesh_sublist[i]
            #print(vertices.shape)
            pc_coor = pc_normal_sublist[i][:, :3]
            normals = pc_normal_sublist[i][:, 3:]
            bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            #print(bounds.shape)
            pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
            pc_coor = pc_coor / (bounds[1] - bounds[0]).max()
            pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995  # input should be from -1 to 1
            assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
            normalized_pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
            normalized_pc_normal_list.append(normalized_pc_normal)
        normalized_pc_normal_array = np.array(normalized_pc_normal_list)
        face_coord = self.face_coods[idx]
        mask = self.mask[idx]

        return normalized_pc_normal_array, face_coord, mask

    def save_dataset(self, save_path):
        self.shuffle_dataset()
    # 获取当前时间并格式化为字符串
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建以时间为名称的文件夹
        save_dir = os.path.join(save_path, timestamp)
        os.makedirs(save_dir, exist_ok=True)
    
    # 生成保存文件的路径
        file_path = os.path.join(save_dir, 'dataset.pkl')
        '''
        mesh_info_list = []
        for mesh in self.return_mesh_list:
            if isinstance(mesh, trimesh.Trimesh):
                mesh_info = mesh.to_dict()  # 使用trimesh的to_dict方法转换为可序列化的字典
                mesh_info_list.append(mesh_info)
            else:
                print("wrong mesh ob!")
                # 处理其他类型的mesh对象
                pass        
        '''
    # 将数据集保存为pickle文件
        with open(file_path, 'wb') as f:
            pickle.dump({
                'pc_normal_list': self.pc_normal_list,
                'return_mesh_list': self.return_mesh_list,
                #'point_feature': self.point_feature,
                'face_coods': self.face_coods,
                'mask': self.mask,
                'downsampletimes': self.downsampletimes
            }, f)
    
        print(f"Dataset saved to {file_path}")

    def info(self):
        print("self.pc_normal_list len: ",len(self.pc_normal_list))
        print("self.return_mesh_list len: ",len(self.return_mesh_list))
        #print("self.point_feature shape: ",self.point_feature.shape)
        print("self.face_coods shape: ",self.face_coods.shape)
        print("self.mask shape: ",self.mask.shape)
    
    def shuffle_dataset(self):
        # 获取当前数据大小
        size = len(self.pc_normal_list)
        downsampletimes = len(self.pc_normal_list[0])
        
        # 生成随机索引数组
        indices = np.random.permutation(size * downsampletimes)
        
        # 重新映射到 (size, downsampletimes) 的二维索引
        shuffled_indices = [(idx // downsampletimes, idx % downsampletimes) for idx in indices]
        
        # 创建新的数据结构用于存储打乱后的数据
        new_pc_normal_list = []
        new_return_mesh_list = []
        new_face_coods = []
        new_mask = []
        
        for i, j in shuffled_indices:
            # 添加打乱后的数据
            new_pc_normal_list.append(self.pc_normal_list[i][j])
            new_return_mesh_list.append(self.return_mesh_list[i][j])
            new_face_coods.append(self.face_coods[i, j])
            new_mask.append(self.mask[i, j])
        
        # 重新组织为 size x downsampletimes 的嵌套结构
        self.pc_normal_list = [
            new_pc_normal_list[i * downsampletimes:(i + 1) * downsampletimes] for i in range(size)
        ]
        self.return_mesh_list = [
            new_return_mesh_list[i * downsampletimes:(i + 1) * downsampletimes] for i in range(size)
        ]
        self.face_coods = np.array(new_face_coods).reshape(size, downsampletimes, *self.face_coods.shape[2:])
        self.mask = np.array(new_mask).reshape(size, downsampletimes, *self.mask.shape[2:])

        print("data shuffled completed!")

        return

    @staticmethod
    def load_dataset(load_path):
        # 从pickle文件加载数据集
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        dataset = GLBDataset.__new__(GLBDataset)  # 使用__new__方法创建实例
        dataset.pc_normal_list = data['pc_normal_list']
        #dataset.return_mesh_list = [o3d.geometry.TriangleMesh.load(mesh_info) for mesh_info in data['return_mesh_list']]
        #dataset.return_mesh_list = [trimesh.Trimesh(**mesh_info) for mesh_info in data['return_mesh_list']]
        dataset.return_mesh_list = data['return_mesh_list']
        #dataset.point_feature = data['point_feature']
        dataset.face_coods = data['face_coods']
        dataset.mask = data['mask']
        dataset.downsampletimes = data['downsampletimes']
        return dataset
    


