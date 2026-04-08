# 直接训练VQ-VAE处理IMU数据 - Transformer版本的SimVQ实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ===================== 数据处理工具 =====================
def normalize_imu_data(imu_data):
    """Z-score标准化IMU数据"""
    return (imu_data - imu_data.mean()) / (imu_data.std() + 1e-8)

def parse_label_file(label_file):
    """解析标注文件"""
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    actions = []
    for hand in label_data:
        for action in label_data[hand]:
            start_time = action['开始时间']
            end_time = action['结束时间']
            description = action['描述']
            start_seconds = convert_time_to_seconds(start_time)
            end_seconds = convert_time_to_seconds(end_time)
            actions.append((start_seconds, end_seconds, description))
    return actions

def convert_time_to_seconds(time_str):
    """时间转秒"""
    t = datetime.strptime(time_str, "%H:%M")
    return t.minute * 60 + t.second

# ===================== Transformer组件 =====================

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerVQEncoder(nn.Module):
    """Transformer VQ-VAE编码器"""
    def __init__(self, input_dim=24, d_model=256, latent_dim=64, 
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影到潜在维度
        self.output_proj = nn.Linear(d_model, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x, mask=None):
        """
        x: (B, T, input_dim) - IMU序列
        mask: (B, T) - 可选的padding mask
        返回: (B, T, latent_dim) - 编码序列
        """
        # 输入投影
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.input_norm(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 创建padding mask用于attention
        if mask is not None:
            # mask shape: (B, T) -> 转换为Transformer需要的格式
            # True表示要mask的位置（padding）
            mask = ~mask  # 反转mask (True变False, False变True)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (B, T, d_model)
        
        # 输出投影
        z_e = self.output_proj(x)  # (B, T, latent_dim)
        z_e = self.output_norm(z_e)
        
        return z_e


class ImprovedSimVQQuantizer(nn.Module):
    """改进的SimVQ矢量量化层 - 增加数值稳定性"""
    def __init__(self, num_embeddings=256, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # 潜在基础B - 使用更小的初始化范围
        scale = 1.0 / math.sqrt(num_embeddings)
        self.register_buffer('latent_basis', torch.randn(num_embeddings, embedding_dim) * scale)
        
        # 可学习的线性变换层W - 使用Xavier初始化
        self.linear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=0.1)  # 小的gain值以稳定训练
        
        # EMA更新（可选，用于追踪码本使用情况）
        self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('ema_w', self.latent_basis.clone())
        
    def get_codebook(self):
        """通过线性层生成码本"""
        # 应用线性变换并归一化以保持数值稳定
        codebook = self.linear_layer(self.latent_basis)
        # L2归一化码本向量
        codebook = F.normalize(codebook, dim=-1) * math.sqrt(self.embedding_dim)
        return codebook
    
    def forward(self, z_e):
        """
        z_e: (B, T, D) - 编码器输出
        """
        B, T, D = z_e.shape
        
        # Flatten
        z_e_flat = z_e.reshape(-1, D)
        
        # 获取码本
        codebook = self.get_codebook()  # (num_embeddings, D)
        
        # 计算L2距离
        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True) 
            - 2 * z_e_flat @ codebook.t() 
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        
        # 找最近的码本向量
        indices = distances.argmin(dim=1)
        
        # 获取量化后的向量
        z_q_flat = codebook[indices]
        
        # Reshape
        z_q = z_q_flat.view(B, T, D)
        indices = indices.view(B, T)
        
        # 使用EMA更新追踪码本使用情况（不影响梯度）
        if self.training:
            encodings = F.one_hot(indices.flatten(), self.num_embeddings).float()
            ema_cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * ema_cluster_size
        
        # VQ损失 - 添加小的epsilon以避免NaN
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # 计算困惑度和码本利用率
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=[0, 1])
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 计算实际使用的码本数量
        used_codes = len(torch.unique(indices))
        codebook_usage = used_codes / self.num_embeddings
        
        return z_q, vq_loss, indices, perplexity, codebook_usage


class TransformerVQDecoder(nn.Module):
    """Transformer VQ-VAE解码器"""
    def __init__(self, latent_dim=64, d_model=256, output_dim=24,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer解码器层（使用编码器结构，因为这是自回归重建）
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, z_q, mask=None):
        """
        z_q: (B, T, latent_dim) - 量化后的表示
        mask: (B, T) - 可选的padding mask
        返回: (B, T, output_dim) - 重建的IMU序列
        """
        # 输入投影
        x = self.input_proj(z_q)  # (B, T, d_model)
        x = self.input_norm(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 创建padding mask
        if mask is not None:
            mask = ~mask  # 反转mask
        
        # Transformer解码
        x = self.transformer_decoder(x, src_key_padding_mask=mask)  # (B, T, d_model)
        
        # 输出投影
        x_recon = self.output_proj(x)  # (B, T, output_dim)
        
        return x_recon


class TransformerSimVQVAE(nn.Module):
    """Transformer版本的SimVQ-VAE模型"""
    def __init__(self, input_dim=24, d_model=256, latent_dim=64, 
                 num_embeddings=256, commitment_cost=0.25,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerVQEncoder(
            input_dim=input_dim,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.quantizer = ImprovedSimVQQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost
        )
        
        self.decoder = TransformerVQDecoder(
            latent_dim=latent_dim,
            d_model=d_model,
            output_dim=input_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        """
        x: (B, T, 24) - 原始IMU序列
        mask: (B, T) - 可选的padding mask
        """
        # 编码
        z_e = self.encoder(x, mask)
        
        # 量化（使用SimVQ）
        z_q, vq_loss, indices, perplexity, codebook_usage = self.quantizer(z_e)
        
        # 解码
        x_recon = self.decoder(z_q, mask)
        
        return x_recon, vq_loss, perplexity, codebook_usage


# ===================== 数据集 =====================

class IMUDataset(Dataset):
    def __init__(self, imu_dir, label_dir, user_num, scene_num):
        self.imu_dir = imu_dir
        self.label_dir = label_dir
        self.user_num = user_num
        self.scene_num = scene_num
        self.data = self.prepare_data()
        
    def prepare_data(self):
        data = []
        
        for i in range(1, self.user_num + 1):
            for j in range(1, self.scene_num + 1):
                # 读取4个传感器的IMU数据
                imu_features_all_sensors = []
                valid_scene = True
                
                for sensor_id in range(4):
                    imu_file = os.path.join(self.imu_dir, f"{i}", f"WT{sensor_id}", f"{j}_WT{sensor_id}.csv")
                    if not os.path.exists(imu_file):
                        valid_scene = False
                        break
                    
                    imu_data = pd.read_csv(imu_file)
                    imu_data_normalized = normalize_imu_data(imu_data[
                        ['hand_acc_6g_x', 'hand_acc_6g_y', 'hand_acc_6g_z',
                         'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z']])
                    imu_features = imu_data_normalized.to_numpy()
                    imu_features_all_sensors.append(imu_features)
                
                if not valid_scene:
                    continue
                
                # 合并传感器数据 (T, 24)
                imu_features_combined = np.concatenate(imu_features_all_sensors, axis=-1)
                
                # 读取标签文件
                label_file = os.path.join(self.label_dir, f"{i}", f"{j}.txt")
                if not os.path.exists(label_file):
                    continue
                    
                actions = parse_label_file(label_file)
                
                # 将每个动作片段加入数据集
                for start, end, description in actions:
                    # 简化处理：直接使用索引作为时间
                    imu_segment = imu_features_combined[start*100:end*100]  # 假设100Hz采样率
                    if len(imu_segment) > 10:  # 只保留足够长的片段
                        data.append(imu_segment)
        
        print(f"Loaded {len(data)} IMU segments")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float()


def collate_fn(batch):
    """批处理函数 - 支持mask"""
    # Padding到相同长度
    imu_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in batch])
    
    # 创建mask (True表示有效位置，False表示padding)
    max_len = imu_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    return imu_padded, lengths, mask


# ===================== 训练函数 =====================

def train_transformer_simvqvae(model, dataloader, optimizer, scheduler=None, epochs=50):
    """训练Transformer SimVQ-VAE"""
    model.train()
    
    best_loss = float('inf')
    patience = 0
    max_patience = 15  # Transformer通常需要更多patience
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        total_codebook_usage = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (imu_batch, lengths, mask) in enumerate(pbar):
            imu_batch = imu_batch.to(device)
            mask = mask.to(device)
            
            # 前向传播
            imu_recon, vq_loss, perplexity, codebook_usage = model(imu_batch, mask)
            
            # 只计算非padding位置的重建损失
            mask_expanded = mask.unsqueeze(-1).expand_as(imu_batch)
            masked_imu = imu_batch * mask_expanded
            masked_recon = imu_recon * mask_expanded
            
            # 重建损失（只计算有效位置）
            recon_loss = F.mse_loss(masked_recon, masked_imu, reduction='sum') / mask.sum()
            
            # 总损失
            loss = recon_loss + 0.1 * vq_loss
            
            # 检查NaN
            if torch.isnan(loss):
                print(f"\nNaN detected at batch {batch_idx}, epoch {epoch+1}")
                print(f"  Recon loss: {recon_loss.item()}")
                print(f"  VQ loss: {vq_loss.item()}")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            total_codebook_usage += codebook_usage
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'vq': f'{vq_loss.item():.4f}',
                'ppl': f'{perplexity.item():.1f}',
                'usage': f'{codebook_usage:.1%}'
            })
        
        # Epoch统计
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        avg_codebook_usage = total_codebook_usage / num_batches
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
              f"VQ: {avg_vq_loss:.4f}, Perplexity: {avg_perplexity:.1f}, "
              f"Codebook Usage: {avg_codebook_usage:.1%}")
        
        # 学习率调度
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_transformer_simvq.pth')
            print(f"  Best model saved (loss: {best_loss:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_transformer_simvq.pth')
    print("\nModel saved as 'final_transformer_simvq.pth'")
    
    # 打印最终的码本统计
    with torch.no_grad():
        codebook = model.quantizer.get_codebook()
        print(f"\nFinal codebook stats:")
        print(f"  Codebook shape: {codebook.shape}")
        print(f"  Codebook norm mean: {codebook.norm(dim=-1).mean():.4f}")
        print(f"  Codebook norm std: {codebook.norm(dim=-1).std():.4f}")
        
        # 打印EMA统计
        ema_usage = model.quantizer.ema_cluster_size
        print(f"  EMA cluster sizes - min: {ema_usage.min():.2f}, max: {ema_usage.max():.2f}")
        print(f"  Active codes (EMA > 1): {(ema_usage > 1).sum().item()} / {len(ema_usage)}")


# ===================== 主函数 =====================

def main():
    # 参数设置
    imu_dir = '../../data/4_IMU_time'
    label_dir = '../../data/4_label_translate'
    user_num = 20
    scene_num = 20
    
    # 训练参数
    batch_size = 16
    learning_rate = 1e-4  # Transformer通常需要更小的学习率
    epochs = 150
    warmup_epochs = 10
    
    print("Loading data...")
    dataset = IMUDataset(imu_dir, label_dir, user_num, scene_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print("Creating Transformer SimVQ-VAE model...")
    model = TransformerSimVQVAE(
        input_dim=24,
        d_model=256,
        latent_dim=64,
        num_embeddings=256,
        commitment_cost=0.25,
        nhead=8,  # 注意力头数
        num_layers=4,  # Transformer层数
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Encoder: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  - Quantizer: {sum(p.numel() for p in model.quantizer.parameters()):,}")
    print(f"  - Decoder: {sum(p.numel() for p in model.decoder.parameters()):,}")
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器 - 使用warmup + cosine annealing
    # 或者使用更简单的调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=len(dataloader) * 10,  # 每10个epoch重启
        T_mult=2,  # 每次重启周期翻倍
        eta_min=1e-6  # 最小学习率
    )
    
    print("\nStarting training with Transformer SimVQ...")
    print(f"Training for {epochs} epochs with {warmup_epochs} warmup epochs")
    train_transformer_simvqvae(model, dataloader, optimizer, scheduler, epochs)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()