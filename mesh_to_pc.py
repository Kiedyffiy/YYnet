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

def get_facecood(mesh, normalize=True):
    if normalize:
        vertices, center, scale_factor = normalize_vertices(mesh.vertices)
    else:
        vertices = mesh.vertices
        center = None
        scale_factor = None
    face_coords = vertices[mesh.faces]
    face_coord_tensor = torch.tensor(face_coords, dtype=torch.float32)
    print(f"Face coordinates shape: {face_coord_tensor.shape}")
    return face_coord_tensor


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


