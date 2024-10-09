import os
import torch
from torch.utils.data import Dataset, DataLoader
from mesh_to_pc import process_shapenet_models
import pickle
from datetime import datetime
import open3d as o3d
import trimesh

class GLBDataset(Dataset):
    def __init__(self, datapath, k = 800 , marching_cubes=True, sample_num=4096, save_path=None):
        # 过滤掉面数超过k的模型并生成数据集
        self.point_feature, self.face_coods, self.mask = process_shapenet_models(
            datapath, k=k, marching_cubes=marching_cubes, sample_num=sample_num
        )
        
        # 可选：保存处理后的数据集
        if save_path is not None:
            self.save_dataset(save_path)

    def __len__(self):
        return self.point_feature.shape[0]

    def __getitem__(self, idx):
        #pc_normal = self.pc_normal_list[idx]
        #mesh = self.return_mesh_list[idx] #减小内存占用
        #print(mesh.shape)
        pointfeature = self.point_feature[idx]
        face_coord = self.face_coods[idx]
        mask = self.mask[idx]

        return pointfeature, face_coord, mask

    def save_dataset(self, save_path):
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
                #'pc_normal_list': self.pc_normal_list,
                #'return_mesh_list': self.return_mesh_list,
                'point_feature': self.point_feature,
                'face_coods': self.face_coods,
                'mask': self.mask
            }, f)
    
        print(f"Dataset saved to {file_path}")

    def info(self):
        #print("self.pc_normal_list len: ",len(self.pc_normal_list))
        #print("self.return_mesh_list len: ",len(self.return_mesh_list))
        print("self.point_feature shape: ",self.point_feature.shape)
        print("self.face_coods shape: ",self.face_coods.shape)
        print("self.mask shape: ",self.mask.shape)

    @staticmethod
    def load_dataset(load_path):
        # 从pickle文件加载数据集
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        dataset = GLBDataset.__new__(GLBDataset)  # 使用__new__方法创建实例
        #dataset.pc_normal_list = data['pc_normal_list']
        #dataset.return_mesh_list = [o3d.geometry.TriangleMesh.load(mesh_info) for mesh_info in data['return_mesh_list']]
        #dataset.return_mesh_list = [trimesh.Trimesh(**mesh_info) for mesh_info in data['return_mesh_list']]
        #dataset.return_mesh_list = data['return_mesh_list']
        dataset.point_feature = data['point_feature']
        dataset.face_coods = data['face_coods']
        dataset.mask = data['mask']
        
        return dataset
    


