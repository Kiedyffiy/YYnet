import trimesh
import os

# 指定ShapeNet数据集的路径
shape_net_path = '/root/data/ShapeNetCore.v2/03001627'

# 统计面数小于800的模型数量
count = 0

# 遍历每个模型文件
for model_name in os.listdir(shape_net_path):
    model_path = os.path.join(shape_net_path, model_name, 'models', 'model_normalized.obj')
    if os.path.exists(model_path):
        try:
            # 加载模型
            scene = trimesh.load(model_path)
            
            # 合并所有子网格
            if isinstance(scene, trimesh.Scene):
                mesh = scene.dump(concatenate=True)  # 合并所有子网格
            else:
                mesh = scene
            
            # 计算面数
            if len(mesh.faces) < 800:
                count += 1
        except Exception as e:
            print(f'处理模型 {model_path} 时出错: {e}')

print(f'面数小于800的模型数量: {count}')


