# 对比重构的IMU与原始IMU数据 - Transformer版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 假设您的Transformer训练代码文件名为train_transformer_vqvae.py
# 导入Transformer版本的模型和数据集
# 注意：您需要根据实际的Transformer训练代码文件名和类名调整这里的导入
from train_transformer_vqvae import TransformerSimVQVAE, IMUDataset, collate_fn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ===================== 对比分析函数 =====================

def load_transformer_model(model_path='best_transformer_simvq.pth', model_config=None):
    """加载训练好的Transformer模型
    
    Args:
        model_path: 模型权重文件路径
        model_config: 模型配置字典，包含模型初始化参数
    """
    # 默认配置 - 您需要根据实际的Transformer模型参数调整
    if model_config is None:
        model_config = {
            'input_dim': 24,
            'd_model': 256,
            'latent_dim': 64,
            'num_embeddings': 256,
            'commitment_cost': 0.25,
            'nhead': 8,  # Transformer特有参数
            'num_layers': 4,  # Transformer特有参数
            'dropout': 0.1   # Transformer特有参数
        }
    
    # 创建Transformer模型实例
    model = TransformerSimVQVAE(**model_config).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理可能的checkpoint格式差异
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path} (with checkpoint dict)")
        # 如果checkpoint中包含配置信息，可以打印出来
        if 'config' in checkpoint:
            print(f"Model config from checkpoint: {checkpoint['config']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")
    
    model.eval()
    return model


def get_reconstruction(model, dataloader, num_samples=5):
    """获取原始和重构的IMU数据"""
    original_data = []
    reconstructed_data = []
    latent_codes = []  # 存储潜在编码，用于后续分析
    
    model.eval()
    with torch.no_grad():
        for i, (imu_batch, lengths, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            imu_batch = imu_batch.to(device)
            
            # 获取重构和潜在表示
            outputs = model(imu_batch)
            
            # 处理不同的输出格式
            if isinstance(outputs, tuple):
                if len(outputs) >= 3:
                    imu_recon, vq_loss, perplexity = outputs[:3]
                    # 如果模型还返回了潜在编码
                    if len(outputs) >= 4:
                        z_q = outputs[3]
                        latent_codes.append(z_q.cpu().numpy())
                else:
                    imu_recon = outputs[0]
            else:
                imu_recon = outputs
            
            original_data.append(imu_batch.cpu().numpy())
            reconstructed_data.append(imu_recon.cpu().numpy())
    
    return original_data, reconstructed_data, latent_codes


def plot_comparison(original, reconstructed, sample_idx=0, save_path=None):
    """
    绘制原始IMU数据和重构数据的对比
    每个传感器的每个通道单独绘制
    """
    # 获取单个样本
    orig = original[sample_idx]  # (T, 24)
    recon = reconstructed[sample_idx]  # (T, 24)
    
    # 创建4x6的子图（4个传感器，每个6个通道）
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.suptitle('IMU Reconstruction Comparison (Transformer VQ-VAE)', fontsize=16)
    
    sensor_names = ['WT0', 'WT1', 'WT2', 'WT3']
    channel_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    for sensor_id in range(4):
        for channel_id in range(6):
            ax = axes[sensor_id, channel_id]
            
            # 计算通道索引
            ch_idx = sensor_id * 6 + channel_id
            
            # 绘制原始和重构信号
            time_steps = range(len(orig))
            ax.plot(time_steps, orig[:, ch_idx], 'b-', label='Original', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, recon[:, ch_idx], 'r--', label='Reconstructed', alpha=0.7, linewidth=1.5)
            
            # 计算该通道的MSE
            mse = np.mean((orig[:, ch_idx] - recon[:, ch_idx]) ** 2)
            
            ax.set_title(f'{sensor_names[sensor_id]} - {channel_names[channel_id]}\nMSE: {mse:.4f}', fontsize=10)
            ax.set_xlabel('Time Steps', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_error_distribution(original_list, reconstructed_list, save_path=None):
    """绘制重构误差分布"""
    # 由于序列长度不同，我们需要分别处理每个batch
    channel_errors_list = []
    all_errors_flat = []  # 用于绘制误差直方图
    
    for orig_batch, recon_batch in zip(original_list, reconstructed_list):
        # 计算误差 (batch_size, T, 24)
        errors = (orig_batch - recon_batch) ** 2
        # 计算这个batch中每个通道的平均误差
        batch_channel_errors = np.mean(errors, axis=(0, 1))  # (24,)
        channel_errors_list.append(batch_channel_errors)
        # 将误差展平用于直方图
        all_errors_flat.extend(errors.flatten())
    
    # 计算所有batch的平均通道误差
    channel_errors = np.mean(channel_errors_list, axis=0)  # (24,)
    
    # 绘制误差分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 按通道的误差条形图
    ax1 = axes[0, 0]
    ax1.bar(range(24), channel_errors, color='steelblue')
    ax1.set_xlabel('Channel Index')
    ax1.set_ylabel('Average MSE')
    ax1.set_title('Reconstruction Error by Channel (Transformer)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 按传感器分组的误差
    ax2 = axes[0, 1]
    sensor_errors = channel_errors.reshape(4, 6)
    sensor_names = ['WT0', 'WT1', 'WT2', 'WT3']
    
    x = np.arange(6)
    width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i in range(4):
        ax2.bar(x + i*width, sensor_errors[i], width, label=sensor_names[i], color=colors[i])
    
    ax2.set_xlabel('Channel Type')
    ax2.set_ylabel('Average MSE')
    ax2.set_title('Reconstruction Error by Sensor (Transformer)')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分布直方图
    ax3 = axes[1, 0]
    ax3.hist(all_errors_flat, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax3.set_xlabel('MSE Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Reconstruction Errors (Transformer)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 按传感器类型（加速度计 vs 陀螺仪）的平均误差
    ax4 = axes[1, 1]
    acc_errors = []
    gyro_errors = []
    for i in range(4):
        acc_errors.extend(sensor_errors[i, :3])
        gyro_errors.extend(sensor_errors[i, 3:])
    
    bp = ax4.boxplot([acc_errors, gyro_errors], labels=['Accelerometer', 'Gyroscope'],
                      patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen']):
        patch.set_facecolor(color)
    ax4.set_ylabel('MSE')
    ax4.set_title('Error Distribution by Sensor Type (Transformer)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Transformer VQ-VAE Reconstruction Error Analysis', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to {save_path}")
    
    plt.show()
    
    return channel_errors


def plot_temporal_comparison(original, reconstructed, sample_idx=0, channels=[0, 6, 12, 18], save_path=None):
    """
    绘制特定通道的时序对比（每个传感器选一个加速度通道）
    """
    orig = original[sample_idx]  # (T, 24)
    recon = reconstructed[sample_idx]  # (T, 24)
    
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 3*len(channels)))
    fig.suptitle('Temporal Comparison of Selected Channels (Transformer VQ-VAE)', fontsize=14)
    
    sensor_names = ['WT0 Acc_X', 'WT1 Acc_X', 'WT2 Acc_X', 'WT3 Acc_X']
    
    for idx, ch in enumerate(channels):
        ax = axes[idx] if len(channels) > 1 else axes
        
        time_steps = range(len(orig))
        
        # 绘制原始信号
        ax.plot(time_steps, orig[:, ch], 'b-', label='Original', linewidth=2)
        # 绘制重构信号
        ax.plot(time_steps, recon[:, ch], 'r-', label='Reconstructed', linewidth=1.5, alpha=0.8)
        
        # 绘制误差带
        error = np.abs(orig[:, ch] - recon[:, ch])
        ax.fill_between(time_steps, recon[:, ch] - error, recon[:, ch] + error, 
                        color='red', alpha=0.2, label='Error Band')
        
        # 计算相关系数
        if np.std(orig[:, ch]) > 0 and np.std(recon[:, ch]) > 0:
            corr = np.corrcoef(orig[:, ch], recon[:, ch])[0, 1]
        else:
            corr = 0.0
        mse = np.mean((orig[:, ch] - recon[:, ch]) ** 2)
        
        ax.set_title(f'{sensor_names[idx]} - Correlation: {corr:.3f}, MSE: {mse:.4f}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal comparison saved to {save_path}")
    
    plt.show()


def plot_attention_analysis(model, dataloader, num_samples=3, save_path=None):
    """
    分析Transformer的注意力模式（Transformer特有的分析）
    注意：这个函数假设您的Transformer模型能返回注意力权重
    """
    print("Analyzing attention patterns...")
    
    model.eval()
    attention_weights = []
    
    with torch.no_grad():
        for i, (imu_batch, lengths, mask) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            imu_batch = imu_batch.to(device)
            
            # 尝试获取注意力权重
            # 这需要您的Transformer模型支持返回注意力权重
            try:
                # 假设模型有一个方法可以返回注意力权重
                if hasattr(model, 'get_attention_weights'):
                    attn = model.get_attention_weights(imu_batch)
                    attention_weights.append(attn.cpu().numpy())
                else:
                    print("Model does not support attention weight extraction")
                    return
            except Exception as e:
                print(f"Could not extract attention weights: {e}")
                return
    
    if len(attention_weights) > 0:
        # 可视化第一个样本的注意力模式
        fig, axes = plt.subplots(1, min(3, len(attention_weights)), figsize=(15, 5))
        if len(attention_weights) == 1:
            axes = [axes]
        
        for idx, attn in enumerate(attention_weights[:3]):
            if attn.ndim >= 2:
                # 取第一个头的注意力权重
                attn_map = attn[0] if attn.ndim > 2 else attn
                
                im = axes[idx].imshow(attn_map[:50, :50], cmap='hot', interpolation='nearest')
                axes[idx].set_title(f'Attention Pattern - Sample {idx+1}')
                axes[idx].set_xlabel('Keys')
                axes[idx].set_ylabel('Queries')
                plt.colorbar(im, ax=axes[idx])
        
        plt.suptitle('Transformer Attention Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention analysis saved to {save_path}")
        
        plt.show()


def calculate_metrics(original_list, reconstructed_list):
    """计算重构质量指标"""
    all_mse = []
    all_mae = []
    all_corr = []
    all_snr = []  # 添加信噪比
    
    for orig_batch, recon_batch in zip(original_list, reconstructed_list):
        # MSE
        mse = np.mean((orig_batch - recon_batch) ** 2)
        all_mse.append(mse)
        
        # MAE
        mae = np.mean(np.abs(orig_batch - recon_batch))
        all_mae.append(mae)
        
        # 信噪比 SNR
        signal_power = np.mean(orig_batch ** 2)
        noise_power = np.mean((orig_batch - recon_batch) ** 2)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            all_snr.append(snr)
        
        # 相关系数（平均所有通道）
        batch_corr = []
        for b in range(orig_batch.shape[0]):
            for c in range(orig_batch.shape[2]):
                if np.std(orig_batch[b, :, c]) > 0 and np.std(recon_batch[b, :, c]) > 0:
                    corr = np.corrcoef(orig_batch[b, :, c], recon_batch[b, :, c])[0, 1]
                    if not np.isnan(corr):
                        batch_corr.append(corr)
        
        if batch_corr:
            all_corr.append(np.mean(batch_corr))
    
    metrics = {
        'MSE': float(np.mean(all_mse)),
        'MSE_std': float(np.std(all_mse)),
        'MAE': float(np.mean(all_mae)),
        'MAE_std': float(np.std(all_mae)),
        'Correlation': float(np.mean(all_corr)) if all_corr else 0.0,
        'Correlation_std': float(np.std(all_corr)) if all_corr else 0.0,
        'SNR': float(np.mean(all_snr)) if all_snr else 0.0,
        'SNR_std': float(np.std(all_snr)) if all_snr else 0.0
    }
    
    return metrics



def evaluate_model(model, dataloader, save_metrics=True):
    """全面评估模型性能"""
    print("\nEvaluating Transformer VQ-VAE model performance...")
    
    original_data = []
    reconstructed_data = []
    total_vq_loss = 0
    total_perplexity = 0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for imu_batch, lengths, mask in tqdm(dataloader, desc="Processing batches"):
            imu_batch = imu_batch.to(device)
            
            # 获取重构
            outputs = model(imu_batch)
            
            # 处理不同的输出格式
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                imu_recon, vq_loss, perplexity = outputs[:3]
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
            else:
                imu_recon = outputs[0] if isinstance(outputs, tuple) else outputs
            
            original_data.append(imu_batch.cpu().numpy())
            reconstructed_data.append(imu_recon.cpu().numpy())
            num_batches += 1
    
    # 计算指标
    metrics = calculate_metrics(original_data, reconstructed_data)
    
    # 添加VQ特定指标
    if num_batches > 0:
        metrics['VQ_Loss'] = total_vq_loss / num_batches
        metrics['Perplexity'] = total_perplexity / num_batches
    
    print("\n" + "="*50)
    print("Transformer VQ-VAE Reconstruction Metrics:")
    print("="*50)
    print(f"MSE: {metrics['MSE']:.6f} ± {metrics['MSE_std']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f} ± {metrics['MAE_std']:.6f}")
    print(f"Correlation: {metrics['Correlation']:.4f} ± {metrics['Correlation_std']:.4f}")
    if 'SNR' in metrics:
        print(f"SNR: {metrics['SNR']:.2f} ± {metrics['SNR_std']:.2f} dB")
    if 'VQ_Loss' in metrics:
        print(f"VQ Loss: {metrics['VQ_Loss']:.4f}")
    if 'Perplexity' in metrics:
        print(f"Perplexity: {metrics['Perplexity']:.2f}")
    print("="*50)
    
    # 保存指标
    if save_metrics:
        import json
        metrics_file = 'transformer_vqvae_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_file}")
    
    return original_data, reconstructed_data, metrics


# ===================== 主函数 =====================

def main():
    """主函数 - 执行Transformer VQ-VAE的完整评估"""
    
    # ============ 配置参数 ============
    # 数据路径
    imu_dir = '../../data/4_IMU_time'
    label_dir = '../../data/4_label_translate'
    user_num = 20
    scene_num = 20
    
    # 模型配置
    model_path = 'best_transformer_simvq.pth'  # Transformer模型权重文件
    
    # 如果您的Transformer模型有不同的配置，请在这里修改
    model_config = {
        'input_dim': 24,
        'd_model': 256,
        'latent_dim': 64,
        'num_embeddings': 256,
        'commitment_cost': 0.25,
        'nhead': 8,      # Transformer特有
        'num_layers': 4,     # Transformer特有
        'dropout': 0.1      # Transformer特有
    }
    
    # ============ 加载数据 ============
    print("Loading dataset...")
    dataset = IMUDataset(imu_dir, label_dir, user_num, scene_num)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset size: {len(dataset)} samples")
    
    # ============ 加载模型 ============
    print("\nLoading Transformer VQ-VAE model...")
    try:
        model = load_transformer_model(model_path, model_config)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please ensure you have trained the Transformer model and the weights file exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your model configuration and file path.")
        return
    
    # ============ 全面评估 ============
    original_data, reconstructed_data, metrics = evaluate_model(model, dataloader)
    
    # ============ 可视化分析 ============
    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)
    
    # 1. 详细对比图（所有通道）
    print("\n1. Plotting detailed channel comparison...")
    plot_comparison(original_data[0], reconstructed_data[0], 
                   sample_idx=0, save_path='transformer_comparison_all_channels.png')
    
    # 2. 时序对比（选择特定通道）
    print("\n2. Plotting temporal comparison...")
    plot_temporal_comparison(original_data[0], reconstructed_data[0], 
                            sample_idx=0, save_path='transformer_temporal_comparison.png')
    
    # 3. 误差分布
    print("\n3. Plotting error distribution...")
    channel_errors = plot_error_distribution(original_data[:5], reconstructed_data[:5], 
                                            save_path='transformer_error_distribution.png')
    
    # 4. Transformer特有：注意力分析（如果模型支持）
    print("\n4. Analyzing attention patterns (if supported)...")
    plot_attention_analysis(model, dataloader, num_samples=3, 
                          save_path='transformer_attention_analysis.png')
    
    # ============ 打印详细的通道误差 ============
    print("\n" + "="*50)
    print("Per-channel MSE Analysis:")
    print("="*50)
    
    sensor_names = ['WT0', 'WT1', 'WT2', 'WT3']
    channel_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    # 找出最好和最差的通道
    best_channel_idx = np.argmin(channel_errors)
    worst_channel_idx = np.argmax(channel_errors)
    
    for i in range(24):
        sensor_id = i // 6
        channel_id = i % 6
        error = channel_errors[i]
        
        # 标记最好和最差的通道
        marker = ""
        if i == best_channel_idx:
            marker = " ← Best"
        elif i == worst_channel_idx:
            marker = " ← Worst"
        
        print(f"  {sensor_names[sensor_id]:4s} - {channel_names[channel_id]:7s}: {error:.6f}{marker}")
    
    # 打印汇总统计
    print("\nChannel Error Statistics:")
    print(f"  Mean: {np.mean(channel_errors):.6f}")
    print(f"  Std:  {np.std(channel_errors):.6f}")
    print(f"  Min:  {np.min(channel_errors):.6f} (Channel {best_channel_idx})")
    print(f"  Max:  {np.max(channel_errors):.6f} (Channel {worst_channel_idx})")
    


    
    # ============ 完成 ============
    print("\n" + "="*50)
    print("Analysis complete! All results have been saved.")
    print("="*50)
    print("\nGenerated files:")
    print("  - transformer_comparison_all_channels.png")
    print("  - transformer_temporal_comparison.png")
    print("  - transformer_error_distribution.png")
    print("  - transformer_attention_analysis.png (if supported)")
    print("  - transformer_vqvae_metrics.json")


if __name__ == "__main__":
    main()