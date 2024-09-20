import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from autoencoder import AutoEncoder
from scipy.optimize import linear_sum_assignment #Hungarian算法
#from chamferdist import ChamferDistance #ChamferDistance

class AutoTrainer(nn.Module):
    def __init__(
        self, 
        model : AutoEncoder, 
        dataset, 
        lr=1e-4, 
        epochs=100, 
        batch_size=8, 
        device = torch.device('cuda')
        
    ):
        super().__init__()
        self.model = model.to(device)
        self.dataset = dataset
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        #self.criterion_chamfer = ChamferDistance()

    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists()

        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        path = Path(path)
        assert path.exists()

        self.model.load_state_dict(torch.load(path))

    def forward(self, pc_list, mesh_list, face_coords, mask, noise=None):
        return self.model(pc_list, mesh_list, face_coords, mask, noise)

    #def compute_set_loss(self, pred_triangles, target_triangles):  #nomarl
        #return self.criterion(pred_triangles, target_triangles)
    
    def compute_set_loss(self, pred_triangles, target_triangles):   #Hungarian
        batch_size, num_faces, num_vertices, _ = pred_triangles.shape
        total_loss = 0
        pred_triangles = pred_triangles.to(self.device)
        target_triangles = target_triangles.to(self.device)
        for b in range(batch_size):
            cost_matrix = torch.cdist(pred_triangles[b], target_triangles[b], p=2)  # 计算两组三角形之间的距离矩阵
            cost_matrix = cost_matrix.mean(dim=-1)  # 计算顶点坐标的平均距离

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
    def compute_set_loss_chamfer(self, pred_triangles, target_triangles): #chamfer
        batch_size = pred_triangles.shape[0]

        # 将 pred_triangles 和 target_triangles 展平到 (b * nf, 3, 3) 形状
        pred_triangles = pred_triangles.view(batch_size, -1, 3)
        target_triangles = target_triangles.view(batch_size, -1, 3)

        loss = self.criterion_chamfer(pred_triangles, target_triangles)
        
        return loss
    '''

    def train_step(self, batch):
        pc_list, mesh_list, face_coords, mask = batch
        self.optimizer.zero_grad()
        
        output = self.forward(pc_list, mesh_list, face_coords, mask)
        pred_triangles = self.model.recon2(output['logits'])
        
        # 计算 set loss
        loss = self.compute_set_loss(pred_triangles, face_coords)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self):
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(self.dataloader):.4f}')

