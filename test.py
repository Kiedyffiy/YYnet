from functools import wraps
import numpy as np
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
#from torch_cluster import fps
from timm.models.layers import DropPath
from mesh_to_pc import (
    process_mesh_to_pc,
    process_shapenet_models,
    calc_feature
)
from pathlib import Path
import trimesh
from MeshAnything.miche.michelangelo.models.tsal.inference_utils import extract_geometry
from functools import partial
from MeshAnything.miche.encode import load_model
from autoencoder import AutoEncoder
from dataread import GLBDataset
from autotrainer import AutoTrainer
from datetime import datetime

#shapenet_data_dir = '/root/src/MeshAnything-main/examples' 

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

shapenet_data_dir = '/root/data/ShapeNetCore.v2/03001627'
save_path1 = '/root/data/YYnetcheck'

save_path1 = os.path.join(save_path1,timestamp)

save_path2 = '/root/data/YYnetdata'

save_path3 = Path(save_path1) / f"trainres.pt"

save_path4 = "/root/data/YYnetrebuild"

pkl_file_path =  '/root/data/YYnetdata/20241009_010626/dataset.pkl'
print("start process loaddata!")

dataset = GLBDataset.load_dataset(pkl_file_path)
#dataset = GLBDataset(datapath = shapenet_data_dir , k = 800 , save_path = save_path2)

#pc_normal_list, return_mesh_list, face_coods, mask = process_shapenet_models(shapenet_data_dir, marching_cubes=True, sample_num=4096)

dataset.info()

print("process_shapenet_finished!")
model = AutoEncoder()

model.to(model.device)

#model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

batch_size = 16
epochs = 100

trainer = AutoTrainer(model=model, dataset=dataset, batch_size=batch_size, epochs=epochs,savepath=save_path1,modelsavepath=save_path4)

trainer.load("/root/data/YYnetcheck/20241011_140856/trainres.pt")

#trainer.train()

#trainer.save(save_path3)

trainer.evaluate()

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