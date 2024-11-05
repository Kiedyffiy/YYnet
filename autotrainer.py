import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator
from autoencoder import (
    AutoEncoder,
    gaussian_blur_1d
)
from scipy.optimize import linear_sum_assignment  # Hungarian算法
from tqdm.auto import tqdm
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import random_split
from rebuild import (
    export_models,
    export_cpmodels
)
from einops import rearrange, repeat, reduce, pack, unpack
import os
import math
class AutoTrainer(nn.Module):
    def __init__(
        self, 
        model: AutoEncoder, 
        dataset, 
        test_dataset,
        lr=1e-4, 
        epochs=100, 
        batch_size=16, 
        test_size = 32,
        bin_smooth_blur_sigma = 0.4,
        discretize_size = 128,
        savepath = None,
        modelsavepath = None,
        device = torch.device('cuda'),
    ):
        super().__init__()
        # 使用Accelerator处理
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision="fp16") #,gradient_accumulation_steps=4
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma
        self.discretize_size = discretize_size
        self.model = model
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        #self.train_size = len(dataset) - test_size
        #self.test_size = test_size
        #self.train_dataset, self.test_dataset = random_split(dataset, [self.train_size, self.test_size])
        self.train_dataset = dataset
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True ,drop_last=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False ,drop_last=True)
        #self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.savepath = Path(savepath)
        self.modelsavepath = modelsavepath
        # 将模型、优化器和数据加载器交给accelerator进行处理 , self.test_dataloader
        self.model, self.optimizer, self.train_dataloader,self.test_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,self.test_dataloader
        )

    def save(self, path, overwrite=True):
        if not isinstance(path, Path):
            path = Path(path)
        # 检查文件是否存在
        if not overwrite and path.exists():
            raise FileExistsError(f"{path} already exists. Use overwrite=True to overwrite.")
        
        path.parent.mkdir(parents=True, exist_ok=True)  # 创建所有缺失的父目录

        if self.accelerator.is_main_process:
            print("main process start saving!")
            torch.save({'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
                        'optimizer_state_dict':self.optimizer.state_dict()}, path)
            print("save pt successfully!")
        self.accelerator.wait_for_everyone()

    def load(self, path):
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        
        # 确保 path 是一个 Path 对象，并且文件存在
        if not isinstance(path, Path):
            path = Path(path)
        assert path.exists()
        
        # 使用 map_location='cpu' 将 checkpoint 加载到 CPU
        checkpoint = torch.load(path, map_location='cpu')
        
        # 将模型移到原始设备 (通过 self.accelerator.unwrap_model)
        self.model = self.accelerator.unwrap_model(self.model)
        
        # 从 CPU 加载模型权重和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复多 GPU 模型（如果之前是分布式训练，accelerate 会自动同步到多个设备）
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        
        # 手动调整学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        print("Load checkpoint successfully!")
        print("self.model.tokens: ", self.accelerator.unwrap_model(self.model).tokens)

    def forward(self, point_feature, mask):
        return self.model(point_feature, mask)
    
    def compute_masked_loss(self, pred_triangles, target_triangles, mask):
        # 获取张量的形状
        batch_size, num_downsample, max_nf, _, _, _ = pred_triangles.shape

        # 1. 将 pred_triangles 和 target_triangles 展平为形状 [batch_size * num_downsample * max_nf, 3, 3]
        pred_triangles_flat = pred_triangles.contiguous().view(batch_size * num_downsample * max_nf, 3, 3, 128)
        target_triangles_flat = target_triangles.contiguous().view(batch_size * num_downsample * max_nf, 3, 3, 128)

        # 2. 将 mask 展平为形状 [batch_size * num_downsample * max_nf]
        mask_flat = mask.contiguous().view(batch_size * num_downsample * max_nf)

        # 3. 使用 mask 选择有效的三角形
        pred_triangles_valid = pred_triangles_flat[mask_flat]
        target_triangles_valid = target_triangles_flat[mask_flat]

        # 4. 计算有效三角形的损失
        loss = self.criterion(pred_triangles_valid, target_triangles_valid)

        return loss

    def compute_dis_loss(self, pred, target, mask):

        recon_losses = (-target * pred).sum(dim = 1)

        mask1 = rearrange(mask,'b nd nf -> b (nd nf)')
        mask1 = repeat(mask1,'b n ->b (n r)', r = 3 * 3)

        recon_loss = recon_losses[mask1].mean()

        return recon_loss
    
    def train_step(self, batch):
        point_feature, face_coords, mask = batch
        #print("point_feature shape:",point_feature.shape)
        #point_feature = torch.tensor(point_feature)
        #face_coords = torch.tensor(face_coords)
        #mask = torch.tensor(mask)
        dst = mask.shape[1]
        total_loss = 0.0
        total_size = 0
        for i in range(dst):
            point_feature1 = point_feature[:,0,:,:]
            face_coords1 = face_coords[:,i:i+1,:,:,:]
            mask1 = mask[:,i:i+1,:]
            self.optimizer.zero_grad()

            pred_triangles = self.forward(point_feature1, mask1)

            # 计算set loss
            #print("pred_triangles.requires_grad:", pred_triangles.requires_grad)
            target1 = self.calc_target(face_coords1,pred_triangles)
            #print("pred_triangles: ",pred_triangles)
            #print("target1: ",target1)
            loss = self.compute_dis_loss(pred_triangles, target1, mask1)
            
            self.accelerator.backward(loss)  # 使用accelerator处理反向传播

            self.optimizer.step()

            total_loss += loss.item()
            total_size += pred_triangles.shape[0]
        #print("loss: ",loss.item())
        return total_loss/dst ,total_size

    def train(self):
        print("len of self.train_dataloader.dataset: ",len(self.train_dataloader.dataset))
        print("len of self.train_dataloader: ",len(self.train_dataloader))
        print("self.train_dataloader.batchsize: ",self.train_dataloader.batch_sampler.batch_size)
        for epoch in tqdm(range(self.epochs), disable=not self.accelerator.is_local_main_process):
            self.model.train()
            epoch_loss = 0.0
            total_batchSize = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch [{epoch+1}/{self.epochs}]", disable=not self.accelerator.is_local_main_process)

            for batch in progress_bar:
                loss,batchSize = self.train_step(batch)
                epoch_loss +=  (loss * batchSize)
                total_batchSize += batchSize
                progress_bar.set_postfix({'Loss': loss})

            epoch_loss_tensor = torch.tensor(epoch_loss, device=self.device)
            total_batchSize_tensor = torch.tensor(total_batchSize, device=self.device)
            gather_epoch_loss = self.accelerator.gather(epoch_loss_tensor)
            gather_total_batchSize = self.accelerator.gather(total_batchSize_tensor)

            if self.accelerator.is_main_process:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {gather_epoch_loss.sum().item()/gather_total_batchSize.sum().item():.4f}, Total_Sap: {gather_total_batchSize.sum().item()}')
            #print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.train_dataloader):.4f}')
                    # 每10个epoch保存一次checkpoint
            #if (epoch + 1) % 5 == 0:
                #self.evaluate()
            #if (epoch + 1) % 10 == 0:
                #checkpoint_path = self.savepath / f"checkpoint_epoch_{epoch+1}_loss_{gather_epoch_loss.sum().item()/gather_total_batchSize.sum().item():.4f}.pt"
                #print("save checkpoint!")
                #self.save(checkpoint_path)

    
    def evaluate(self):
        self.model.eval()  # 评估模式
        total_loss = 0.0
        total_samples = 0
        print("main process start eval !")
        # 禁用梯度计算
        with torch.no_grad():
            for batch in self.test_dataloader:
                point_feature, face_coords, mask = batch
                point_feature = point_feature[:,0,:,:]
                face_coords = face_coords[:,0:1,:,:,:]
                mask = mask[:,0:1,:]
                # 前向传播
                pred_triangles = self.forward(point_feature, mask)

                # 计算损失
                loss = self.compute_masked_loss(pred_triangles, face_coords, mask)

                # 累积损失和样本数量
                total_loss += loss.item() * pred_triangles.shape[0]  # 乘以batch中的样本数
                total_samples += pred_triangles.shape[0]

                #重建模型
                current_device = torch.cuda.current_device()
                save_dir = os.path.join(self.modelsavepath,f"v4")
                #export_cpmodels(pred_triangles,face_coords,mask,save_dir)
                #print(f"export finish! GPU: {current_device}")

        # 计算平均损失
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        total_samples_tensor = torch.tensor(total_samples, device=self.device)
        gather_total_loss = self.accelerator.gather(total_loss_tensor)
        gather_total_samples = self.accelerator.gather(total_samples_tensor)

        if self.accelerator.is_main_process:
            avg_loss = gather_total_loss.sum().item() / gather_total_samples.sum().item()
            print("Here is testres_loss : ",avg_loss ,"total_samples: ",gather_total_samples.sum().item())
        self.accelerator.wait_for_everyone()
        return 
    
    def calc_target(self, face_coordinates, pred_log_prob):
        '''
        face_coordinates [b, nd, nf, 3, 3] - ground truth continuous coordinates
        pred_log_prob  Shape: [b, 128 ,(nd, nf, 3, 3)]
        '''
        target_discretized = ((face_coordinates - face_coordinates.min()) / 
                              (face_coordinates.max() - face_coordinates.min()) * (self.discretize_size - 1)).long()
        target_discretized = rearrange(target_discretized,'b ... -> b 1 (...)')
        target_one_hot = torch.zeros_like(pred_log_prob).scatter_(1, target_discretized, 1.)

        # Apply Gaussian blur if bin_smooth_blur_sigma >= 0
        if self.bin_smooth_blur_sigma > 0:
            target_one_hot = gaussian_blur_1d(target_one_hot, sigma=self.bin_smooth_blur_sigma)        
        return target_one_hot #Shape: [b, 128 ,(nd, nf, 3, 3)]
    '''
    def compute_set_loss(self, pred_triangles, target_triangles):  # Hungarian
        batch_size, num_faces, num_vertices, _ = pred_triangles.shape
        total_loss = 0

        for b in range(batch_size):
            # 计算两组三角形之间的距离矩阵
            cost_matrix = torch.cdist(pred_triangles[b], target_triangles[b], p=2)
            cost_matrix = cost_matrix.mean(dim=-1)

            # 使用 Hungarian 算法进行二分匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

            # 按照匹配结果重新排列 pred_triangles
            matched_pred_triangles = pred_triangles[b][row_indices]
            matched_target_triangles = target_triangles[b][col_indices]

            # 计算匹配后的损失
            loss = self.criterion(matched_pred_triangles, matched_target_triangles)
            total_loss += loss

        return total_loss / batch_size
    '''
    '''
    def train_step(self, batch):
        point_feature, face_coords, mask = batch

        face_coords = face_coords[:,0:1,:,:,:]
        mask = mask[:,0:1,:]
        self.optimizer.zero_grad()

        pred_triangles = self.forward(point_feature, mask)

            # 计算set loss
        loss = self.compute_masked_loss(pred_triangles, face_coords, mask)
        self.accelerator.backward(loss)  # 使用accelerator处理反向传播

        self.optimizer.step()

        #print("loss: ",loss.item())
        return loss.item(),pred_triangles.shape[0]
    '''
    '''
    def compute_masked_loss(self, pred_triangles, target_triangles, mask):
        batch_size, num_downsample, max_nf, _, _ = pred_triangles.shape

        # 将 pred_triangles 中的无效面填充为 0
        pred_triangles_masked = pred_triangles.clone()  # 复制一份以避免修改原始数据
        pred_triangles_masked[~mask] = 0  # 将无效面填充为 0

        # 计算损失
        loss = self.criterion(pred_triangles_masked, target_triangles)

        return loss
    '''