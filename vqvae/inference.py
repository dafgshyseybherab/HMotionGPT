"""
推理脚本：读取训练好的模型，将原始IMU数据传入模型进行推理，保存码本输出（indices和z_q）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ===================== 数据处理工具 =====================
def normalize_imu_data(imu_data):
    """Z-score标准化IMU数据"""
    return (imu_data - imu_data.mean()) / (imu_data.std() + 1e-8)


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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerVQEncoder(nn.Module):
    """Transformer VQ-VAE编码器"""
    def __init__(self, input_dim=24, d_model=256, latent_dim=64, 
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
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
        
        self.output_proj = nn.Linear(d_model, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        x = self.pos_encoding(x)
        
        if mask is not None:
            mask = ~mask
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        z_e = self.output_proj(x)
        z_e = self.output_norm(z_e)
        
        return z_e


class ImprovedSimVQQuantizer(nn.Module):
    """改进的SimVQ矢量量化层"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        scale = 1.0 / math.sqrt(num_embeddings)
        self.register_buffer('latent_basis', torch.randn(num_embeddings, embedding_dim) * scale)
        
        self.linear_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=0.1)
        
        self.register_buffer('ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('ema_w', self.latent_basis.clone())
        
    def get_codebook(self):
        codebook = self.linear_layer(self.latent_basis)
        codebook = F.normalize(codebook, dim=-1) * math.sqrt(self.embedding_dim)
        return codebook
    
    def forward(self, z_e):
        B, T, D = z_e.shape
        
        z_e_flat = z_e.reshape(-1, D)
        
        codebook = self.get_codebook()
        
        distances = (
            z_e_flat.pow(2).sum(dim=1, keepdim=True) 
            - 2 * z_e_flat @ codebook.t() 
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        
        indices = distances.argmin(dim=1)
        
        z_q_flat = codebook[indices]
        
        z_q = z_q_flat.view(B, T, D)
        indices = indices.view(B, T)
        
        if self.training:
            encodings = F.one_hot(indices.flatten(), self.num_embeddings).float()
            ema_cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * ema_cluster_size
        
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        z_q = z_e + (z_q - z_e).detach()
        
        encodings = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=[0, 1])
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        used_codes = len(torch.unique(indices))
        codebook_usage = used_codes / self.num_embeddings
        
        return z_q, vq_loss, indices, perplexity, codebook_usage


class TransformerVQDecoder(nn.Module):
    """Transformer VQ-VAE解码器"""
    def __init__(self, latent_dim=64, d_model=256, output_dim=24,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
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
        
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, z_q, mask=None):
        x = self.input_proj(z_q)
        x = self.input_norm(x)
        
        x = self.pos_encoding(x)
        
        if mask is not None:
            mask = ~mask
        
        x = self.transformer_decoder(x, src_key_padding_mask=mask)
        
        x_recon = self.output_proj(x)
        
        return x_recon


class TransformerSimVQVAE(nn.Module):
    """Transformer版本的SimVQ-VAE模型"""
    def __init__(self, input_dim=24, d_model=256, latent_dim=64, 
                 num_embeddings=512, commitment_cost=0.25,
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
        z_e = self.encoder(x, mask)
        z_q, vq_loss, indices, perplexity, codebook_usage = self.quantizer(z_e)
        x_recon = self.decoder(z_q, mask)
        
        return x_recon, z_q, indices, vq_loss, perplexity, codebook_usage


# ===================== IMU数据集 =====================

class IMUDataset(Dataset):
    """IMU数据集，只读取CSV文件中的IMU数据"""
    def __init__(self, imu_dir, user_num, scene_num):
        self.imu_dir = imu_dir
        self.user_num = user_num
        self.scene_num = scene_num
        self.data = self.prepare_data()
        
    def prepare_data(self):
        data = []
        
        for user_id in range(1, self.user_num + 1):
            for scene_id in range(1, self.scene_num + 1):
                # 读取4个传感器的IMU数据
                imu_features_all_sensors = []
                valid_scene = True
                
                for sensor_id in range(4):
                    imu_file = os.path.join(self.imu_dir, f"{user_id}", f"WT{sensor_id}", f"{scene_id}_WT{sensor_id}.csv")
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
                
                data.append({
                    'imu': imu_features_combined,
                    'user_id': user_id,
                    'scene_id': scene_id
                })
        
        print(f"Loaded {len(data)} IMU sequences")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'imu': torch.tensor(item['imu']).float(),
            'user_id': item['user_id'],
            'scene_id': item['scene_id']
        }


def collate_fn(batch):
    """批处理函数"""
    imu_list = [item['imu'] for item in batch]
    user_ids = [item['user_id'] for item in batch]
    scene_ids = [item['scene_id'] for item in batch]
    
    # Padding到相同长度
    imu_padded = pad_sequence(imu_list, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(seq) for seq in imu_list])
    
    # 创建mask (True表示有效位置，False表示padding)
    max_len = imu_padded.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    return imu_padded, lengths, mask, user_ids, scene_ids


# ===================== 推理和保存函数 =====================

def inference_and_save_codebook(model_path, imu_dir, user_num, scene_num, 
                                output_dir='./codebook_output', batch_size=16):
    """
    推理并保存码本输出
    
    参数：
        model_path: 训练好的模型路径
        imu_dir: IMU数据目录
        user_num: 用户数量
        scene_num: 场景数量
        output_dir: 输出目录
        batch_size: 批处理大小
    
    保存内容：
        - indices: 码本索引 (T,)
        - z_q: 量化后的表示 (T, latent_dim)
        - reconstruction: 重构的IMU数据 (T, 24)
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print("Loading model...")
    model = TransformerSimVQVAE(
        input_dim=24,
        d_model=256,
        latent_dim=64,
        num_embeddings=256,
        commitment_cost=0.25,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    # 加载预训练权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
    
    # 切换到评估模式
    model.eval()
    
    # 准备数据
    print("Preparing dataset...")
    dataset = IMUDataset(imu_dir, user_num, scene_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 推理
    print("Starting inference and saving codebook outputs...")
    total_saved = 0
    
    with torch.no_grad():
        for batch_idx, (imu_batch, lengths, mask, user_ids, scene_ids) in enumerate(tqdm(dataloader, desc="Processing batches")):
            imu_batch = imu_batch.to(device)
            mask = mask.to(device)
            
            # 前向传播
            x_recon, z_q, indices, vq_loss, perplexity, codebook_usage = model(imu_batch, mask)
            
            # 处理每个样本
            imu_batch_cpu = imu_batch.detach().cpu().numpy()
            x_recon_cpu = x_recon.detach().cpu().numpy()
            z_q_cpu = z_q.detach().cpu().numpy()
            indices_cpu = indices.detach().cpu().numpy()
            mask_cpu = mask.cpu().numpy()
            
            batch_size_actual = imu_batch_cpu.shape[0]
            
            for i in range(batch_size_actual):
                # 获取有效长度
                valid_len = mask_cpu[i].sum()
                
                user_id = user_ids[i]
                scene_id = scene_ids[i]
                
                # 创建按原文件夹结构的输出路径
                user_dir = os.path.join(output_dir, f"user_{user_id}")
                scene_dir = os.path.join(user_dir, f"scene_{scene_id}")
                os.makedirs(scene_dir, exist_ok=True)
                
                # 提取有效数据（移除padding）
                indices_item = indices_cpu[i][:valid_len]  # (T,)
                z_q_item = z_q_cpu[i][:valid_len]  # (T, latent_dim)
                imu_original = imu_batch_cpu[i][:valid_len]  # (T, 24)
                imu_recon = x_recon_cpu[i][:valid_len]  # (T, 24)
                
                # 保存码本索引
                np.save(os.path.join(scene_dir, 'indices.npy'), indices_item)
                
                # 保存量化后的表示z_q
                np.save(os.path.join(scene_dir, 'z_q.npy'), z_q_item)
                
                # 保存原始IMU数据
                np.save(os.path.join(scene_dir, 'imu_original.npy'), imu_original)
                
                # 保存重构后的IMU数据
                np.save(os.path.join(scene_dir, 'imu_reconstruction.npy'), imu_recon)
                
                # 计算重构误差
                recon_error = np.mean((imu_original - imu_recon) ** 2)
                
                total_saved += 1
    
    print(f"\n✓ Successfully processed {total_saved} sequences")
    print(f"  Output directory: {output_dir}")
    print(f"  Directory structure: output_dir/user_X/scene_Y/")
    print(f"  Saved files in each scene directory:")
    print(f"    - indices.npy: 码本索引 (T,)")
    print(f"    - z_q.npy: 量化表示 (T, 64)")
    print(f"    - imu_original.npy: 原始IMU (T, 24)")
    print(f"    - imu_reconstruction.npy: 重构IMU (T, 24)")
    
    return total_saved


# ===================== 主函数 =====================

def main():
    # 参数配置
    model_path = './best_transformer_simvq.pth'  # 训练好的模型路径
    imu_dir = '../../data/4_IMU_time'              # IMU数据目录
    user_num = 20
    scene_num = 20
    
    # 输出配置
    output_dir = './codebook_output'
    batch_size = 16
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first.")
        return
    
    # 执行推理并保存
    total = inference_and_save_codebook(
        model_path=model_path,
        imu_dir=imu_dir,
        user_num=user_num,
        scene_num=scene_num,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()