import mesh2sdf.core
import numpy as np
import skimage.measure
import trimesh
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

# 加载ShapeNet模型
def load_shapenet_model(file_path):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        # 如果是Scene对象，合并所有几何体为一个Trimesh对象
        mesh = mesh.dump(concatenate=True)
    return mesh

def normalize_vertices(vertices, scale=0.9):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale

def export_to_watertight(normalized_mesh, octree_depth: int = 7):
    """
        Convert the non-watertight mesh to watertight.

        Args:
            input_path (str): normalized path
            octree_depth (int):

        Returns:
            mesh(trimesh.Trimesh): watertight mesh

        """
    size = 2 ** octree_depth
    level = 2 / size

    scaled_vertices, to_orig_center, to_orig_scale = normalize_vertices(normalized_mesh.vertices)

    sdf = mesh2sdf.core.compute(scaled_vertices, normalized_mesh.faces, size=size)

    vertices, faces, normals, _ = skimage.measure.marching_cubes(np.abs(sdf), level)


    # watertight mesh
    vertices = vertices / size * 2 - 1 # -1 to 1
    vertices = vertices / to_orig_scale + to_orig_center
    # vertices = vertices / to_orig_scale + to_orig_center
    mesh = trimesh.Trimesh(vertices, faces, normals=normals)
    #print("len of in face ",len(faces))
    #face_coord = torch.tensor(vertices[faces], dtype=torch.float32)  # 'nf nvf c'
    #print("face_coord_shape: ",face_coord.shape)
    return mesh  #, face_coord

def sort_vertices_by_zyx(vertex):
    """按照 z-y-x 顺序对单个顶点进行排序"""
    return vertex[2], vertex[1], vertex[0]  # z, y, x

def cyclic_permute(face):
    """将 face 中的顶点进行环形重排，使得最小的索引在第一个位置"""
    min_idx = torch.argmin(face, dim=0)  # 找到最小顶点的索引
    return torch.roll(face, shifts=-min_idx.item(), dims=0)  # 根据最小索引进行环形重排

def reorder_faces(vertices, faces):
    """
    按照 MeshGPT 的方法，对顶点和面片进行排序。
    vertices: 顶点坐标张量，形状为 [num_vertices, 3]，其中 3 表示 (x, y, z) 坐标。
    faces: 面片张量，形状为 [num_faces, 3]，其中 3 表示顶点索引。
    """
    
    # 去重顶点并按 z-y-x 顺序排序
    seen = OrderedDict()
    for point in vertices:
        key = tuple(point.tolist())
        if key not in seen:
            #print(key,"first appear!")
            seen[key] = point

    
    unique_vertices = list(seen.values())
    #print("unique_vertices.shape: ",len(unique_vertices))
    #print(unique_vertices)
    sorted_vertices = sorted(unique_vertices, key=sort_vertices_by_zyx)
    
    # 创建顶点映射
    vertices_as_tuples = [tuple(v.tolist()) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v.tolist()) for v in sorted_vertices]
    
    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples)
                  for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples)
                  if vertex_tuple == sorted_vertex_tuple}
    
    # 根据新顶点索引重排面片
    reindexed_faces = torch.tensor([[vertex_map[face[0].item()], vertex_map[face[1].item()], vertex_map[face[2].item()]] for face in faces])
    #print("reindexed_faces: ",reindexed_faces.shape)
    # 对每个面片内部的顶点进行排序
    reindexed_faces = torch.stack([cyclic_permute(face) for face in reindexed_faces])
    
    # 按最低顶点的索引顺序排列面片
    sorted_faces = reindexed_faces[torch.argsort(reindexed_faces[:, 0])]
    
    return torch.stack(sorted_vertices), sorted_faces

def sort_mesh_with_mask(tensor, mask):
    """
    对 batchsize, downsampletimes, face, vertices, coords 的 tensor 进行排序，仅排序有效面片。
    tensor: 形状为 [batchsize, num_of_downsampletimes, num_faces, num_vertices_per_face (3), coords (3)]
    mask: 有效面片的 mask，形状为 [batchsize, num_of_downsampletimes, max_nf]，有效面片为 True。
    
    返回: 排序后的 tensor。
    """
    #print("tensor.shape: ",tensor.shape)
    #print("mask.shape: ",mask.shape)
    batchsize, nd, nf, nvf, c = tensor.shape
    sorted_tensors = []
    
    print("start sort!!")

    for b in tqdm(range(batchsize), desc="Sorting batches"):
        downsampled_tensors = []
        for i in range(nd):
            face_coords = tensor[b, i]  # 获取当前 downsample 的 face 坐标
            current_mask = mask[b, i]   # 获取当前 downsample 对应的 mask
            
            # 获取有效面片的索引（mask 为 True 的部分）
            valid_indices = torch.nonzero(current_mask, as_tuple=False).squeeze(1)
            
            if valid_indices.numel() == 0:  # 如果没有有效的面片，直接跳过
                downsampled_tensors.append(face_coords)
                continue

            # 只对有效的面片进行排序
            valid_face_coords = face_coords[valid_indices]
            vertices = valid_face_coords.reshape(-1, c)  # 把有效顶点展开成 [num_vertices, coords] 形状
            #print("vertices.shape: ",vertices.shape)
            faces = torch.arange(vertices.shape[0]).reshape(len(valid_indices), nvf)  # 生成对应的 face 索引
            #print("faces.shape: ",faces.shape)
            # 排序有效面片
            sorted_vertices, sorted_faces = reorder_faces(vertices, faces)
            
            # 根据重新排序的面片顺序，重建 face_coords
            sorted_face_coords = sorted_vertices[sorted_faces].reshape(len(valid_indices), nvf, c)
            
            # 将排序后的面片填回到 face_coords 中，无效的面片保持不变
            sorted_full_face_coords = face_coords.clone()
            sorted_full_face_coords[valid_indices] = sorted_face_coords
            
            downsampled_tensors.append(sorted_full_face_coords)
        
        sorted_tensors.append(torch.stack(downsampled_tensors, dim=0))

    print("end sort!!")

    return torch.stack(sorted_tensors, dim=0)


def pad_face_coords_sample(all_face_coods):
    """
    对同一个 mesh 的不同降采样版本的 face_coords 进行填充，确保形状一致，并生成 mask。
    
    参数:
    - all_face_coods: list，里面包含多个不同降采样版本的 face_coords，每个 face_coords 的形状为 [num of face, num of vertice, coord]。
    
    返回:
    - padded_facecoords_list: list，填充后的 face_coords，形状为 [num of face, num of vertice, coord]。
    - mask_list: list，mask，标记有效的面，形状为 [num of face]。
    """
    
    # 找出最大的 faces 数量
    max_nf = max(face_coords.shape[0] for face_coords in all_face_coods)
    nvf, c = all_face_coods[0].shape[1], all_face_coods[0].shape[2]  # nvf和c是相同的
    
    padded_facecoords_list = []
    mask_list = []

    # 对每个 face_coords 进行填充
    for face_coords in all_face_coods:
        nf = face_coords.shape[0]
        
        # 初始化填充后的 face_coords 和 mask
        padded_face_coords = np.zeros((max_nf, nvf, c), dtype=np.float32)
        mask = np.zeros((max_nf,), dtype=np.bool_)
        
        # 填充 face_coords
        padded_face_coords[:nf] = face_coords
        mask[:nf] = True  # 标记有效的面
        
        # 将填充后的 face_coords 和 mask 添加到列表中
        padded_facecoords_list.append(padded_face_coords)
        mask_list.append(mask)
    
    return padded_facecoords_list, mask_list

def get_facecood(mesh, normalize=True):
    if normalize:
        vertices, center, scale_factor = normalize_vertices(mesh.vertices)
    else:
        vertices = mesh.vertices
        center = None
        scale_factor = None
    face_coords = vertices[mesh.faces]
    #face_coord_tensor = torch.tensor(face_coords, dtype=torch.float32)
    #print(f"Face coordinates shape: {face_coord_tensor.shape}")
    return face_coords
    #return face_coord_tensor


def simplify_mesh(mesh, target_faces):
    """简化 mesh 并返回降采样后的 face_coord"""
    #print("target_faces: ",target_faces)
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces) #simplify_quadric_decimation
    #print(simplified_mesh.faces.shape)
    face_coord = get_facecood(simplified_mesh)
    return face_coord

def process_mesh_to_pc(mesh_list, marching_cubes=False, sample_num=4096 ,decrement = 0.05 ,stoppercent = 0.5):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    face_cood_list = []

    for mesh in tqdm(mesh_list, desc="Processing meshes"):
        # 提取原始 face_coord
        original_face_coord = get_facecood(mesh)
        all_face_coods = [original_face_coord]

        # 检查是否需要进行 marching cubes 操作
        if marching_cubes:
            mesh2 = export_to_watertight(mesh)
            #print("MC over!")
            return_mesh_list.append(mesh2)
        else:
            mesh2 = mesh
            return_mesh_list.append(mesh2)
        

        
        num_faces = len(mesh.faces)
        # 获取逐次降采样的版本（从 100% 到 50%，每次减少 5%）
        #decrement = 0.05  # 每次减少5%
        current_faces = num_faces
        times = (1 - stoppercent)/decrement
        turns = 0
        while turns < times:
            current_faces = int(current_faces - num_faces * decrement)
            downsampled_face_coord = simplify_mesh(mesh, current_faces)
            all_face_coods.append(downsampled_face_coord)
            turns += 1  

        '''
        # 获取降采样的版本（面数减半和减少到四分之一）
        half_faces = num_faces // 2
        quarter_faces = num_faces // 4

        # 获取降采样后的 face_coord 并拼接
        half_face_coord = simplify_mesh(mesh, half_faces)
        quarter_face_coord = simplify_mesh(mesh, quarter_faces)

        # 将原始和降采样后的 face_coord 进行拼接
        all_face_coods.append(half_face_coord)
        all_face_coods.append(quarter_face_coord)
        '''
        # 拼接成新 tensor，形状为 [降采样数量，面片数量，每个面顶点数，顶点坐标]
        #combined_face_coord = torch.stack(all_face_coods, dim=0)  #[num of downsampletimes，num of face，num of vertice，coord]
        #combined_face_coord = np.stack(all_face_coods, axis=0)
        #print("all_face_coods.len: ",len(all_face_coods))
        face_cood_list.append(all_face_coods)

        # 采样点云并计算法向量
        points, face_idx = mesh2.sample(sample_num, return_index=True)
        normals = mesh2.face_normals[face_idx]
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)

        #print("Process mesh success")

    return pc_normal_list, return_mesh_list, face_cood_list

'''
def process_mesh_to_pc(mesh_list, marching_cubes = False, sample_num = 4096):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    face_cood_list = []
    for mesh in mesh_list:

        #vertices = mesh.vertices  # 顶点数组，形状为 [num_vertices, 3]
        #faces = mesh.faces  # 面数组，形状为 [num_faces, num_vertices_per_face]
        face_coord = get_facecood(mesh)
        face_cood_list.append(face_coord)

        if marching_cubes:
            mesh= export_to_watertight(mesh)   # , face_cood 
            #face_cood_list.append(face_cood)
            print("MC over!")
        return_mesh_list.append(mesh)
        
        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]

        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)
        print("process mesh success")

        #face_coord = vertices[faces]  # 形状为 [num_faces, num_vertices_per_face, 3]
        #face_coord_tensor = torch.tensor(face_coord, dtype=torch.float32)
        #face_cood_list.append(face_coord_tensor)
        #print("process facecoods success")

    return pc_normal_list, return_mesh_list, face_cood_list
'''
'''
def pad_face_coord_list(face_coord_list):
    # 找出最大的nf
    max_nf = max(face_coord.shape[0] for face_coord in face_coord_list)
    nvf, c = face_coord_list[0].shape[1], face_coord_list[0].shape[2]  # nvf和c是相同的
    
    # 初始化填充后的tensor和mask
    batch_size = len(face_coord_list)
    padded_face_coord = torch.zeros((batch_size, max_nf, nvf, c), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_nf), dtype=torch.bool)

    for i, face_coord in enumerate(face_coord_list):
        nf = face_coord.shape[0]
        padded_face_coord[i, :nf] = face_coord  # 填充
        mask[i, :nf] = True  # 仅标记有效的face部分

    return padded_face_coord, mask
'''
def pad_face_coord_list(face_coord_list):
    # 找出最大的降采样次数和最大的nf
    print("pad_face_coord start!!")
    num_of_downsampletimes = max(len(downsampled_list) for downsampled_list in face_coord_list)
    max_nf = max(max(face_coord.shape[0] for face_coord in downsampled_list) for downsampled_list in face_coord_list)
    nvf, c = face_coord_list[0][0].shape[1], face_coord_list[0][0].shape[2]  # nvf 和 c 是相同的
    #print("num_of_downsampletimes: ",num_of_downsampletimes)
    #print("max_nf: ",max_nf)
    # 初始化填充后的tensor和mask
    batch_size = len(face_coord_list)
    padded_face_coord = torch.zeros((batch_size, num_of_downsampletimes, max_nf, nvf, c), dtype=torch.float32)
    mask = torch.zeros((batch_size, num_of_downsampletimes, max_nf), dtype=torch.bool)

    for i, downsampled_list in enumerate(face_coord_list):
        for j, face_coord in enumerate(downsampled_list):
            nf = face_coord.shape[0]
            # 将np数组转换为torch tensor并填充
            face_coord_tensor = torch.tensor(face_coord, dtype=torch.float32)
            padded_face_coord[i, j, :nf] = face_coord_tensor
            mask[i, j, :nf] = True  # 仅标记有效的face部分

    print("pad_face_coord finish!!")

    return sort_mesh_with_mask(padded_face_coord,mask), mask

# 处理ShapeNet模型并生成点云
# todo 适应数据结构
'''
def process_shapenet_models(data_dir,k = 800, marching_cubes=False, sample_num=4096):
    mesh_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.obj') or file.endswith('.off') or file.endswith('.glb'):
                file_path = os.path.join(root, file)
                mesh = load_shapenet_model(file_path)

                # 检查面数是否超过阈值k
                if len(mesh.faces) > k:
                    print(f"Skipping {file} as it has {len(mesh.faces)} faces, which exceeds the threshold of {k}.")
                    continue 

                mesh_list.append(mesh)
    
    pc_normal_list, return_mesh_list,face_cood_list = process_mesh_to_pc(mesh_list, marching_cubes = marching_cubes, sample_num = sample_num)

    padded_face_coord, mask = pad_face_coord_list(face_cood_list)

    return pc_normal_list, return_mesh_list, padded_face_coord, mask
'''

def process_shapenet_models(data_dir,k = 800, marching_cubes=False, sample_num=4096):
    mesh_list = []
    for model_name in os.listdir(data_dir):
        model_path = os.path.join(data_dir, model_name, 'models', 'model_normalized.obj')
        if os.path.exists(model_path):
            if os.path.getsize(model_path) > 1 * 1024 * 1024:
                continue
            try:
                mesh = load_shapenet_model(model_path)

                if len(mesh.faces) > k:
                    #print(f"Skipping {file} as it has {len(mesh.faces)} faces, which exceeds the threshold of {k}.")
                    continue 
                mesh_list.append(mesh)
            except Exception as e:
                print(f'处理模型 {model_path} 时出错: {e}')
    print("Generate meshlist successfully! len of list : ",len(mesh_list))

    pc_normal_list, return_mesh_list,face_cood_list = process_mesh_to_pc(mesh_list, marching_cubes = marching_cubes, sample_num = sample_num)

    padded_face_coord, mask = pad_face_coord_list(face_cood_list)

    return pc_normal_list, return_mesh_list, padded_face_coord, mask


'''
def sort_vertices_by_zyx(vertices):
    """按照 z-y-x 顺序对顶点进行排序"""
    return vertices[:, 2], vertices[:, 1], vertices[:, 0]  # z, y, x

def cyclic_permute(face):
    """将 face 中的顶点进行环形重排，使得最小的索引在第一个位置"""
    min_idx = torch.argmin(face, dim=0)  # 找到最小顶点的索引
    return torch.roll(face, shifts=-min_idx[0].item(), dims=0)  # 根据最小索引进行环形重排

def reorder_faces(vertices, faces):
    """
    按照 MeshGPT 的方法，对顶点和面片进行排序。
    vertices: 顶点坐标张量，形状为 [num_vertices, 3]，其中 3 表示 (x, y, z) 坐标。
    faces: 面片张量，形状为 [num_faces, 3]，其中 3 表示顶点索引。
    """
    
    # 去重顶点并按 z-y-x 顺序排序
    seen = OrderedDict()
    for point in vertices:
        key = tuple(point.tolist())
        if key not in seen:
            seen[key] = point
    
    unique_vertices = list(seen.values())
    sorted_vertices = sorted(unique_vertices, key=sort_vertices_by_zyx)
    
    # 创建顶点映射
    vertices_as_tuples = [tuple(v.tolist()) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v.tolist()) for v in sorted_vertices]
    
    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples)
                  for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples)
                  if vertex_tuple == sorted_vertex_tuple}
    
    # 根据新顶点索引重排面片
    reindexed_faces = torch.tensor([[vertex_map[face[0].item()], vertex_map[face[1].item()], vertex_map[face[2].item()]] for face in faces])
    
    # 对每个面片内部的顶点进行排序
    reindexed_faces = torch.stack([cyclic_permute(face) for face in reindexed_faces])
    
    # 按最低顶点的索引顺序排列面片
    sorted_faces = reindexed_faces[torch.argsort(reindexed_faces[:, 0])]
    
    return torch.stack(sorted_vertices), sorted_faces

def sort_mesh_recon(tensor):  #To do Mask
    """
    对 batchsize, downsampletimes, face, vertices, coords 的 tensor 进行排序。
    tensor: 形状为 [batchsize, num_of_downsampletimes, num_faces, num_vertices_per_face (3), coords (3)]
    
    返回: 排序后的 tensor。
    """
    batchsize, nd, nf, nvf, c = tensor.shape
    sorted_tensors = []
    
    for b in range(batchsize):
        downsampled_tensors = []
        for i in range(nd):
            face_coords = tensor[b, i]  # 获取当前 downsample 的 face 坐标
            vertices = face_coords.reshape(-1, c)  # 把顶点展开成 [num_vertices, coords] 形状
            faces = torch.arange(vertices.shape[0]).reshape(nf, nvf)  # 生成对应的 face 索引

            sorted_vertices, sorted_faces = reorder_faces(vertices, faces)

            # 根据重新排序的面片顺序，重建 face_coords
            sorted_face_coords = sorted_vertices[sorted_faces].reshape(nf, nvf, c)
            downsampled_tensors.append(sorted_face_coords)
        
        sorted_tensors.append(torch.stack(downsampled_tensors, dim=0))
    
    return torch.stack(sorted_tensors, dim=0)

'''