"""
改进的Z_Q和文本标签对齐系统 - Version 2 (Final Enhanced with Labels)
==================================================

功能说明:
1. CN_llm_desc_prompts: 存放JSON元数据文件
   - 文件格式: {instruction, inputs, 对齐描述, 微调描述, 标签}
   - 文件名: u1_s1_右手_L000_0003_0005_info.json (添加用户和场景前缀)
   - inputs: 包含zq, indices, imu三个.npy文件名

2. onehot_tokens: 统一存放所有 .npy 文件
   - 文件名格式: u1_s1_右手_L000_0003_0005_zq.npy (添加用户和场景前缀)

3. 自动压缩: 完成后自动将输出文件夹压缩成ZIP文件，命名为当前目录名_seg.zip

输出目录结构:
输出目录/
├── CN_llm_desc_prompts/
│   ├── user_1/
│   │   ├── scene_1/
│   │   │   ├── u1_s1_左手_L001_0007_0008_info.json
│   │   │   ├── u1_s1_右手_L000_0003_0005_info.json
│   │   │   └── ...
│   │   └── scene_2/
│   └── user_2/
└── onehot_tokens/
    ├── u1_s1_右手_L000_0003_0005_zq.npy
    ├── u1_s1_右手_L000_0003_0005_indices.npy
    ├── u1_s1_右手_L000_0003_0005_imu.npy
    ├── u1_s1_左手_L001_0007_0008_zq.npy
    ├── u1_s1_左手_L001_0007_0008_indices.npy
    ├── u1_s1_左手_L001_0007_0008_imu.npy
    ├── u2_s1_右手_L000_0003_0005_zq.npy
    └── ...
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import re
import argparse
import zipfile
import shutil
from typing import List, Dict, Tuple, Optional


class ZQLabelsAlignerv3Final:
    """Z_Q数据和文本标签对齐类 - 最终增强版本（包含标签字段）"""
    
    def __init__(self, 
                 codebook_dir: str,
                 labels_dir: str,
                 output_dir: str,
                 sampling_rate: Optional[float] = None,
                 auto_estimate_rate: bool = True):
        """
        初始化对齐器
        
        参数:
            codebook_dir: codebook输出目录路径
            labels_dir: 标签文件目录路径
            output_dir: 输出目录路径
            sampling_rate: IMU采样率(Hz)
            auto_estimate_rate: 是否自动估计采样率
        """
        self.codebook_dir = codebook_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.sampling_rate = sampling_rate
        self.auto_estimate_rate = auto_estimate_rate
        
        # 创建输出子目录
        self.desc_prompts_dir = os.path.join(output_dir, "CN_llm_desc_prompts")
        self.tokens_dir = os.path.join(output_dir, "onehot_tokens")
        
        for dir_path in [self.desc_prompts_dir, self.tokens_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.total_aligned = 0
        self.total_failed = 0
    
    @staticmethod
    def parse_timestamp(time_str: str) -> Tuple[Optional[int], Optional[int]]:
        """解析时间戳字符串"""
        try:
            parts = time_str.split('-')
            if len(parts) != 2:
                return None, None
            
            start_str = parts[0].strip()
            end_str = parts[1].strip()
            
            # 解析 MM:SS 格式
            start_parts = start_str.split(':')
            if len(start_parts) == 2:
                start_seconds = int(start_parts[0]) * 60 + int(start_parts[1])
            elif len(start_parts) == 3:  # HH:MM:SS
                start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
            else:
                return None, None
            
            end_parts = end_str.split(':')
            if len(end_parts) == 2:
                end_seconds = int(end_parts[0]) * 60 + int(end_parts[1])
            elif len(end_parts) == 3:  # HH:MM:SS
                end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
            else:
                return None, None
            
            return start_seconds, end_seconds
        except (ValueError, IndexError):
            return None, None
    
    @staticmethod
    def extract_labels_from_json(json_file: str) -> List[Dict]:
        """从JSON标签文件中提取所有标签"""
        try:
            with open(json_file, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            labels = []
            
            # 提取左手标签
            if "左手精细描述" in data and isinstance(data["左手精细描述"], list):
                for idx, item in enumerate(data["左手精细描述"]):
                    timestamp = item.get("时间戳", "")
                    aligned_desc = item.get("对齐描述", "")
                    refined_desc = item.get("微调描述", "")
                    label_tag = item.get("标签", "")  # 添加标签字段提取
                    start_sec, end_sec = ZQLabelsAlignerv3Final.parse_timestamp(timestamp)
                    
                    if start_sec is not None and end_sec is not None:
                        labels.append({
                            'timestamp': timestamp,
                            'hand': '左手',
                            '对齐描述': aligned_desc,
                            '微调描述': refined_desc,
                            '标签': label_tag,  # 添加标签字段
                            'start_sec': start_sec,
                            'end_sec': end_sec,
                            'label_idx': idx
                        })
            
            # 提取右手标签
            if "右手精细描述" in data and isinstance(data["右手精细描述"], list):
                for idx, item in enumerate(data["右手精细描述"]):
                    timestamp = item.get("时间戳", "")
                    aligned_desc = item.get("对齐描述", "")
                    refined_desc = item.get("微调描述", "")
                    label_tag = item.get("标签", "")  # 添加标签字段提取
                    start_sec, end_sec = ZQLabelsAlignerv3Final.parse_timestamp(timestamp)
                    
                    if start_sec is not None and end_sec is not None:
                        labels.append({
                            'timestamp': timestamp,
                            'hand': '右手',
                            '对齐描述': aligned_desc,
                            '微调描述': refined_desc,
                            '标签': label_tag,  # 添加标签字段
                            'start_sec': start_sec,
                            'end_sec': end_sec,
                            'label_idx': idx
                        })
            
            return sorted(labels, key=lambda x: x['start_sec'])
        except Exception as e:
            print(f"  Error parsing {json_file}: {e}")
            return []
    
    def estimate_sampling_rate(self, total_frames: int) -> float:
        """估计IMU采样率"""
        if self.sampling_rate is not None:
            return self.sampling_rate
        # 默认假设30Hz采样率
        return 30.0
    
    def find_label_file(self, user_id: int, scene_id: int) -> Optional[str]:
        """查找用户和场景对应的标签文件"""
        user_dir = os.path.join(self.labels_dir, str(user_id))
        
        if not os.path.exists(user_dir):
            return None
        
        label_files = sorted([f for f in os.listdir(user_dir) if f.endswith('.txt') or f.endswith('.json')])
        
        if scene_id <= len(label_files):
            return os.path.join(user_dir, label_files[scene_id - 1])
        
        return None
    
    def get_instruction(self, hand: str) -> str:
        """根据手获取对应的instruction"""
        if hand == '左手':
            return "给定一个包含左手动作的IMU数据文件，请生成对该片段的自然语言描述，重点说明左手的动作和交互。"
        else:
            return "给定一个包含右手动作的IMU数据文件，请生成对该片段的自然语言描述，重点说明右手的动作和交互。"
    
    def align_single(self, user_id: int, scene_id: int,
                    verbose: bool = True) -> int:
        """对齐单个用户和场景的数据"""
        
        # 读取Z_Q数据
        z_q_path = os.path.join(self.codebook_dir, f"user_{user_id}", 
                                f"scene_{scene_id}", "z_q.npy")
        indices_path = os.path.join(self.codebook_dir, f"user_{user_id}", 
                                   f"scene_{scene_id}", "indices.npy")
        imu_path = os.path.join(self.codebook_dir, f"user_{user_id}", 
                               f"scene_{scene_id}", "imu_original.npy")
        
        if not os.path.exists(z_q_path):
            if verbose:
                print(f"  ✗ Z_Q data not found at {z_q_path}")
            self.total_failed += 1
            return 0
        
        try:
            z_q = np.load(z_q_path)  # (T, 64)
            indices = np.load(indices_path)  # (T,)
            imu_original = np.load(imu_path)  # (T, 24)
        except Exception as e:
            if verbose:
                print(f"  ✗ Error loading data: {e}")
            self.total_failed += 1
            return 0
        
        total_frames = z_q.shape[0]
        current_rate = self.estimate_sampling_rate(total_frames)
        
        if verbose:
            print(f"  Data shape: {z_q.shape}, Sampling rate: {current_rate:.1f} Hz")
        
        # 查找标签文件
        label_file = self.find_label_file(user_id, scene_id)
        
        if not label_file or not os.path.exists(label_file):
            if verbose:
                print(f"  ✗ Label file not found for user {user_id}, scene {scene_id}")
            self.total_failed += 1
            return 0
        
        # 提取标签
        labels = self.extract_labels_from_json(label_file)
        
        if not labels:
            if verbose:
                print(f"  ✗ No labels found in {label_file}")
            self.total_failed += 1
            return 0
        
        # 创建输出目录
        desc_output_path = os.path.join(self.desc_prompts_dir, f"user_{user_id}", f"scene_{scene_id}")
        os.makedirs(desc_output_path, exist_ok=True)
        
        # ===== 生成 DESC_PROMPTS =====
        aligned_count = 0
        
        for label in labels:
            start_sec = label['start_sec']
            end_sec = label['end_sec']
            
            # 转换秒数到帧索引
            start_frame = int(start_sec * current_rate)
            end_frame = int(end_sec * current_rate)
            
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))
            
            if start_frame > end_frame:
                continue
            
            # 截取Z_Q段
            z_q_segment = z_q[start_frame:end_frame+1]
            indices_segment = indices[start_frame:end_frame+1]
            imu_segment = imu_original[start_frame:end_frame+1]
            
            # 生成文件名（添加用户和场景前缀）
            timestamp_str = label['timestamp'].replace('-', '_').replace(':', '')
            label_name = f"u{user_id}_s{scene_id}_{label['hand']}_L{label['label_idx']:03d}_{timestamp_str}"
            
            # 保存.npy文件到onehot_tokens
            npy_file_zq = f"{label_name}_zq.npy"
            npy_file_indices = f"{label_name}_indices.npy"
            npy_file_imu = f"{label_name}_imu.npy"
            
            np.save(os.path.join(self.tokens_dir, npy_file_zq), z_q_segment)
            np.save(os.path.join(self.tokens_dir, npy_file_indices), indices_segment)
            np.save(os.path.join(self.tokens_dir, npy_file_imu), imu_segment)
            
            # 创建描述prompts JSON（增强版本，包含对齐描述、微调描述和标签）
            desc_prompt = {
                "instruction": self.get_instruction(label['hand']),
                "inputs": {
                    "zq": npy_file_zq,
                    "indices": npy_file_indices,
                    "imu": npy_file_imu
                },
                "对齐描述": label['对齐描述'],
                "微调描述": label['微调描述'],
                "标签": label.get('标签', '')  # 添加标签字段
            }
            
            # 保存DESC JSON文件
            info_file = os.path.join(desc_output_path, f"{label_name}_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(desc_prompt, f, ensure_ascii=False, indent=2)
            
            aligned_count += 1
        
        if verbose and aligned_count > 0:
            print(f"  ✓ Aligned {aligned_count} labels")
        
        self.total_aligned += aligned_count
        return aligned_count
    
    def align_batch(self, user_ids: Optional[List[int]] = None,
                   scene_ids: Optional[List[int]] = None) -> Dict:
        """批量对齐多个用户和场景"""
        if user_ids is None:
            user_ids = list(range(1, 21))
        if scene_ids is None:
            scene_ids = list(range(1, 21))
        
        print(f"\n{'='*70}")
        print(f"Starting batch alignment (Version 2 Final Enhanced with Labels)...")
        print(f"Users: {len(user_ids)}, Scenes per user: {len(scene_ids)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        total_scenes = len(user_ids) * len(scene_ids)
        
        with tqdm(total=total_scenes, desc="Overall progress") as pbar:
            for user_id in user_ids:
                print(f"\nUser {user_id}:")
                for scene_id in scene_ids:
                    self.align_single(user_id, scene_id, verbose=False)
                    pbar.update(1)
        
        return {
            'total_aligned': self.total_aligned,
            'total_failed': self.total_failed,
            'output_dir': self.output_dir,
            'desc_prompts_dir': self.desc_prompts_dir,
            'tokens_dir': self.tokens_dir
        }


def zip_directory(source_dir: str, output_zip: str) -> str:
    """
    将目录压缩成ZIP文件
    
    参数:
        source_dir: 要压缩的目录路径
        output_zip: 输出的ZIP文件路径
    
    返回:
        ZIP文件的完整路径
    """
    print(f"\n📦 Compressing directory to ZIP...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_zip}")
    
    # 如果ZIP文件已存在，删除它
    if os.path.exists(output_zip):
        os.remove(output_zip)
        print(f"  ✓ Removed existing ZIP file")
    
    # 创建ZIP文件
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历目录
        total_files = 0
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算相对路径
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)
                total_files += 1
                
                # 每100个文件打印一次进度
                if total_files % 100 == 0:
                    print(f"    Compressed {total_files} files...")
    
    # 获取ZIP文件大小
    zip_size = os.path.getsize(output_zip) / (1024 * 1024)  # 转换为MB
    print(f"  ✓ ZIP created successfully")
    print(f"  Total files: {total_files}")
    print(f"  ZIP size: {zip_size:.2f} MB")
    
    return output_zip


def main():
    parser = argparse.ArgumentParser(
        description='Align Z_Q data with text labels (Version 2 Final Enhanced with Labels)'
    )
    
    parser.add_argument('--codebook-dir', type=str, default='./codebook_output',
                       help='Path to codebook output directory')
    parser.add_argument('--labels-dir', type=str, default='/data01/lwl/ubicomp/data/label_with_tags',
                       help='Path to labels directory')
    parser.add_argument('--output-dir', type=str, default='./zq_labeled_output_v3',
                       help='Path to output directory')
    parser.add_argument('--sampling-rate', type=float, default=None,
                       help='IMU sampling rate in Hz')
    parser.add_argument('--user-ids', type=str, default='1-20',
                       help='User IDs to process (e.g., "1-20" or "1,3,5")')
    parser.add_argument('--scene-ids', type=str, default='1-20',
                       help='Scene IDs to process (e.g., "1-20" or "1,2,3")')
    parser.add_argument('--no-zip', action='store_true',
                       help='Skip creating ZIP file after processing')
    
    args = parser.parse_args()
    
    # 解析用户和场景IDs
    def parse_id_range(id_str):
        if '-' in id_str:
            parts = id_str.split('-')
            return list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            return [int(x) for x in id_str.split(',')]
    
    user_ids = parse_id_range(args.user_ids)
    scene_ids = parse_id_range(args.scene_ids)
    
    # 创建对齐器并执行
    aligner = ZQLabelsAlignerv3Final(
        codebook_dir=args.codebook_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate
    )
    
    result = aligner.align_batch(user_ids, scene_ids)
    
    # 打印结果摘要
    print(f"\n{'='*70}")
    print(f"✓ Alignment Complete (Version 2 Final Enhanced with Labels)!")
    print(f"  Total aligned labels: {result['total_aligned']}")
    print(f"  Failed scenes: {result['total_failed']}")
    print(f"\n📁 Output Structure:")
    print(f"  ├── CN_llm_desc_prompts/user_X/scene_Y/")
    print(f"  │   └── u1_s1_右手_L000_0003_0005_info.json")
    print(f"  │")
    print(f"  └── onehot_tokens/")
    print(f"      ├── u1_s1_右手_L000_0003_0005_zq.npy")
    print(f"      ├── u1_s1_右手_L000_0003_0005_indices.npy")
    print(f"      ├── u1_s1_右手_L000_0003_0005_imu.npy")
    print(f"      ├── u1_s1_左手_L001_0007_0008_zq.npy")
    print(f"      ├── u1_s1_左手_L001_0007_0008_indices.npy")
    print(f"      ├── u1_s1_左手_L001_0007_0008_imu.npy")
    print(f"      ├── u2_s1_右手_L000_0003_0005_zq.npy")
    print(f"      └── ...")
    print(f"\n📄 DESC JSON File Format:")
    print(f"  {{")
    print(f"    \"instruction\": \"给定一个包含左/右手动作的IMU数据文件...\",")
    print(f"    \"inputs\": {{")
    print(f"      \"zq\": \"u1_s1_右手_L000_0003_0005_zq.npy\",")
    print(f"      \"indices\": \"u1_s1_右手_L000_0003_0005_indices.npy\",")
    print(f"      \"imu\": \"u1_s1_右手_L000_0003_0005_imu.npy\"")
    print(f"    }},")
    print(f"    \"对齐描述\": \"左手向上小幅抬起后快速下落，呈弯曲的打字手型...\",")
    print(f"    \"微调描述\": \"左手从键盘上方短暂抬起，随后手指迅速按下特定按键...\",")
    print(f"    \"标签\": \"准备合同签字流程\"")
    print(f"  }}")
    print(f"{'='*70}\n")
    
    # 压缩输出目录
    if not args.no_zip and result['total_aligned'] > 0:
        # 获取当前工作目录名称
        current_dir = os.getcwd()
        current_dir_name = os.path.basename(current_dir)
        
        # 生成ZIP文件名：当前目录名_seg.zip
        zip_filename = f"{current_dir_name}_seg.zip"
        zip_path = os.path.join(current_dir, zip_filename)
        
        print(f"\n📍 Current working directory: {current_dir}")
        print(f"📍 ZIP file will be named: {zip_filename}")
        
        # 压缩目录
        zip_directory(args.output_dir, zip_path)
        
        print(f"\n{'='*70}")
        print(f"✓ All Processing Complete!")
        print(f"  Current directory: {current_dir_name}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  ZIP file: {zip_path}")
        print(f"{'='*70}\n")
    else:
        if args.no_zip:
            print(f"\n✓ Processing complete (ZIP creation skipped)")
        else:
            print(f"\n✓ Processing complete (no data to compress)")


if __name__ == "__main__":
    main()