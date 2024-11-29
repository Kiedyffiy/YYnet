from functools import wraps
import numpy as np
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
#from torch_cluster import fps
from timm.models.layers import DropPath
#from mesh_to_pc import (
#    process_mesh_to_pc,
#    process_shapenet_models,
#    calc_feature
#)
from pathlib import Path
import trimesh
from MeshAnything.miche.michelangelo.models.tsal.inference_utils import extract_geometry
from functools import partial
from MeshAnything.miche.encode import load_model
from autoencoder import AutoEncoder
from dataread import GLBDataset
from autotrainer import AutoTrainer
from datetime import datetime
from torch.utils.data import random_split
import random
#shapenet_data_dir = '/root/src/MeshAnything-main/examples' 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

shapenet_data_dir = '/root/data/ShapeNetCore.v2'

category_list = ["table","bench","tower","lamp","bed","chair"]#

save_path1 = '/root/data/YYnetcheck'

save_path1 = os.path.join(save_path1,timestamp)

save_path2 = '/root/data/YYnetdata'

save_path3 = Path(save_path1) / f"trainres.pt"

save_path4 = "/root/data/YYnetrebuild"

pkl_file_path =  '/root/data/YYnetdata/20241128_165204/dataset.pkl'
#pkl_file_path =  '/root/data/YYnetdata/20241112_115726/dataset.pkl'
print("start process loaddata!")

dataset = GLBDataset.load_dataset(pkl_file_path)
#dataset = GLBDataset(datapath = shapenet_data_dir, category_list = category_list , k = 900 ,marching_cubes=False, save_path = save_path2)

dataset.info()

print(dataset.__len__())

#dataset.shuffle_dataset()

print("process_shapenet_finished!")


'''

dataset1, dataset2 = random_split(dataset, [128, len(dataset) - 128])
subset2_indices = dataset2.indices
torch.save({'subset2_indices': subset2_indices}, '/root/data/YYnetdata/small/subsets_indices3.pth')
print("have save subsets_indices!")

'''



torch.cuda.empty_cache()


checkpoint = torch.load('/root/data/YYnetdata/small/subsets_indices3.pth')
dataset2 = torch.utils.data.Subset(dataset, checkpoint['subset2_indices'])
print("have load subsets_indices!")

# 获取 subset2 的索引
subset2_indices = set(checkpoint['subset2_indices'])
# 获取 dataset 的所有索引
all_indices = set(range(len(dataset)))
# 获取 subset1 的索引 (即不在 subset2 中的索引)
subset1_indices = list(all_indices - subset2_indices)
if len(subset1_indices) < 64:
    raise ValueError("subset1_indices 的样本数量不足 64 个")
subset1_sample_indices = random.sample(subset1_indices, 64)
# 创建 subset1
dataset1 = torch.utils.data.Subset(dataset, subset1_sample_indices)




batch_size = 32
epochs = 500

model = AutoEncoder()
#model.to(model.device)


trainer = AutoTrainer(model=model, dataset=dataset2,test_dataset = dataset1,lr = 1e-5, batch_size=batch_size, epochs=epochs,savepath=save_path1,modelsavepath=save_path4)
trainer.load("/root/data/YYnetcheck/20241127_104440/checkpoint_epoch_220_loss_0.4715.pt")
#trainer.train()

#trainer.save(save_path3)

trainer.evaluate(is_recon = True)

print("Training completed successfully!")





'''

model = AutoEncoder()

model.to(model.device)

batch_size = 1
epochs = 1

trainer = AutoTrainer(model=model, dataset=dataset, batch_size=batch_size, epochs=epochs)
trainer.train()

trainer.save(save_path1)

print("Training completed successfully!")
'''


#print(face_coods.shape)
#print(mask.shape)
#model.to(model.device)

#y = model(pc_normal_list, return_mesh_list, face_coods, mask)

#pred = model.recon2(y['logits'])

#print(pred.shape)

#y = model.encode(pc_normal_list, return_mesh_list, face_cood_list)

#print("test_finish!")




#y = model.encode(pc_normal_list, return_mesh_list)

#print("encode_test_finish!")

'''
point_feature_shape:  torch.Size([2, 257, 768])
pro_point_feature_shape:  torch.Size([2, 257, 768])
x_shape:  torch.Size([2, 257, 768])
encode_test_finish!

'''

#y = model(pc_normal_list, return_mesh_list)
#print(y["logits"].shape)
#print("forward_test_finish!")

'''
context_shape:  torch.Size([2, 257, 768])
noise_shape:  torch.Size([2, 257, 512])
y_shape:  torch.Size([2, 257, 512])
torch.Size([2, 257, 768])
forward_test_finish!

'''
'''
y = model.encode(pc_normal_list, return_mesh_list)
print("encode_test_finish!")

recon = model.recon(y)
print("recon_test_finish!")
'''

'''
mesh_v_f_shape1:  (1081890, 3)
mesh_v_f_shape2:  (2177767, 3)
recon_mesh_shape:  4.296560126800584
recon_test_finish!

'''