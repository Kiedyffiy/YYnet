import os
import trimesh
import numpy as np

# 定义输入和输出目录
input_dir = '/root/src/trypc/testdata'
output_dir = '/root/data/nouse/output'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义OBJ文件列表
obj_files = ['testobj1.glb', 'testobj2.glb']

# 定义降采样百分比列表
decimation_percentages = np.arange(100, 50, -5)

# 测试降采样
for obj_file in obj_files:
    for percentage in decimation_percentages:
        # 加载OBJ文件
        scene = trimesh.load(os.path.join(input_dir, obj_file))
        # 检查是否为Scene对象
        if isinstance(scene, trimesh.Scene):
            # 如果是Scene对象，提取第一个网格
            mesh = scene.dump().pop()
        else:
            mesh = scene
        
        # 计算目标面片数量
        target_faces = int(len(mesh.faces) * (percentage / 100.0))
        
        # 执行降采样s
        simplified_mesh = mesh.simplify_quadric_decimation(target_faces)

        num_faces = len(simplified_mesh.faces)
        
        # 保存降采样结果
        output_file = os.path.join(output_dir, f'{os.path.splitext(obj_file)[0]}_{percentage}_percent.obj')
        simplified_mesh.export(output_file)
        
        print(f'Saved {percentage}% decimated mesh for {obj_file} to {output_file}')
        print(f'This mesh has {num_faces} faces.')

print('Decimation test completed.')


'''
import os
import open3d as o3d
import numpy as np

# 定义输入和输出目录
input_dir = '/root/src/trypc/testdata'
output_dir = '/root/data/nouse/output'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义GLB文件列表
glb_files = ['testobj1.glb', 'testobj2.glb']

# 定义降采样百分比列表
decimation_percentages = np.arange(100, 50, -5)

# 测试降采样
for glb_file in glb_files:
    for percentage in decimation_percentages:
        # 加载GLB文件
        mesh = o3d.io.read_triangle_mesh(os.path.join(input_dir, glb_file))

        # 检查是否成功加载
        if mesh.is_empty():
            print(f"Failed to load mesh: {glb_file}")
            continue
        
        # 计算目标面片数量
        target_faces = int(len(mesh.triangles) * (percentage / 100.0))
        
        # 执行降采样
        simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)

        # 保存降采样结果
        output_file = os.path.join(output_dir, f'{os.path.splitext(glb_file)[0]}_{percentage}_percent.obj')
        o3d.io.write_triangle_mesh(output_file, simplified_mesh)
        
        num_faces = len(simplified_mesh.triangles)
        print(f'Saved {percentage}% decimated mesh for {glb_file} to {output_file}')
        print(f'This mesh has {num_faces} faces.')

print('Decimation test completed.')
'''