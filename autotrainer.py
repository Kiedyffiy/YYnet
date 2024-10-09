import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from accelerate import Accelerator
from autoencoder import AutoEncoder
from scipy.optimize import linear_sum_assignment  # Hungarian算法
from tqdm.auto import tqdm
from accelerate.utils import DistributedDataParallelKwargs

class AutoTrainer(nn.Module):
    def __init__(
        self, 
        model: AutoEncoder, 
        dataset, 
        lr=1e-3, 
        epochs=100, 
        batch_size=8, 
        savepath = None,
        device = torch.device('cuda'),
    ):
        super().__init__()
        # 使用Accelerator处理
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs],gradient_accumulation_steps=4, mixed_precision="fp16")
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.savepath = savepath
        # 将模型、优化器和数据加载器交给accelerator进行处理
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

    def save(self, path, overwrite=True):
        path = Path(path)
        assert overwrite or not path.exists()

        if self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            torch.save(self.accelerator.unwrap_model(self.model).state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(torch.load(path))

    def forward(self, point_feature, face_coords, mask):
        return self.model(point_feature, face_coords, mask)
    
    def compute_masked_loss(self, pred_triangles, target_triangles, mask):
        batch_size, num_downsample, max_nf, _, _ = pred_triangles.shape

        # 将 pred_triangles 中的无效面填充为 0
        pred_triangles_masked = pred_triangles.clone()  # 复制一份以避免修改原始数据
        pred_triangles_masked[~mask] = 0  # 将无效面填充为 0

        # 计算损失
        loss = self.criterion(pred_triangles_masked, target_triangles)

        return loss / batch_size

    def train_step(self, batch):
        point_feature, face_coords, mask = batch
        self.optimizer.zero_grad()

        pred_triangles = self.forward(point_feature, face_coords, mask)
        #pred_triangles = self.model.recon2(output['logits'])

        # 计算set loss
        loss = self.compute_masked_loss(pred_triangles, face_coords, mask)
        self.accelerator.backward(loss)  # 使用accelerator处理反向传播

        self.optimizer.step()

        return loss.item(),pred_triangles.shape[0]

    def train(self):
        for epoch in tqdm(range(self.epochs), disable=not self.accelerator.is_local_main_process):
            epoch_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch [{epoch+1}/{self.epochs}]", disable=not self.accelerator.is_local_main_process)

            for batch in progress_bar:
                loss,batchSize = self.train_step(batch)
                epoch_loss +=  (loss * batchSize)
                progress_bar.set_postfix({'Loss': loss})
            if self.accelerator.is_main_process:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.dataloader):.4f}')
            #print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.dataloader):.4f}')
                    # 每10个epoch保存一次checkpoint
            if (epoch + 1) % 10 == 0 and self.accelerator.is_main_process:
                checkpoint_path = self.savepath / f"checkpoint_epoch_{epoch+1}.pt"
                print("save checkpoint!")
                self.save(checkpoint_path)
    
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