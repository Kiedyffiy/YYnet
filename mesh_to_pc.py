import mesh2sdf.core
import numpy as np
import skimage.measure
import trimesh
import os
import torch

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
    simplified_mesh = mesh.simplify_quadric_decimation(target_faces) #simplify_quadric_decimation
    face_coord = get_facecood(simplified_mesh)
    return face_coord

def process_mesh_to_pc(mesh_list, marching_cubes=False, sample_num=4096 ,decrement = 0.05 ,stoppercent = 0.5):
    # mesh_list : list of trimesh
    pc_normal_list = []
    return_mesh_list = []
    face_cood_list = []

    for mesh in mesh_list:
        # 提取原始 face_coord
        original_face_coord = get_facecood(mesh)
        all_face_coods = [original_face_coord]

        # 检查是否需要进行 marching cubes 操作
        if marching_cubes:
            mesh = export_to_watertight(mesh)
            print("MC over!")
        return_mesh_list.append(mesh)

        
        num_faces = len(mesh.faces)
        # 获取逐次降采样的版本（从 100% 到 50%，每次减少 5%）
        #decrement = 0.05  # 每次减少5%
        current_faces = num_faces

        while current_faces > num_faces * stoppercent:
            current_faces = int(current_faces * (1 - decrement))
            downsampled_face_coord = simplify_mesh(mesh, current_faces)
            all_face_coods.append(downsampled_face_coord)

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
        face_cood_list.append(all_face_coods)

        # 采样点云并计算法向量
        points, face_idx = mesh.sample(sample_num, return_index=True)
        normals = mesh.face_normals[face_idx]
        pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
        pc_normal_list.append(pc_normal)

        print("Process mesh success")

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
    num_of_downsampletimes = max(len(downsampled_list) for downsampled_list in face_coord_list)
    max_nf = max(max(face_coord.shape[0] for face_coord in downsampled_list) for downsampled_list in face_coord_list)
    nvf, c = face_coord_list[0][0].shape[1], face_coord_list[0][0].shape[2]  # nvf 和 c 是相同的

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

    return padded_face_coord, mask

# 处理ShapeNet模型并生成点云
# todo 适应数据结构

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
    
    pc_normal_list, return_mesh_list,face_cood_list = process_mesh_to_pc(mesh_list, marching_cubes, sample_num)

    padded_face_coord, mask = pad_face_coord_list(face_cood_list)

    return pc_normal_list, return_mesh_list, padded_face_coord, mask


