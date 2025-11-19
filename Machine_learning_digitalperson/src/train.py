"""
训练脚本
用于训练口型-音频匹配模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorboard import SummaryWriter
import random

from model import create_model, ContrastiveLoss


class LipSyncDataset(Dataset):
    """口型-音频匹配数据集"""
    
    def __init__(self, data_index_path, sequence_length=5, is_training=True):
        """
        初始化数据集
        Args:
            data_index_path: 数据索引文件路径
            sequence_length: 序列长度
            is_training: 是否为训练模式
        """
        with open(data_index_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        # 过滤掉太短的样本
        self.data_list = [item for item in self.data_list if item['length'] >= sequence_length]
        
        print(f"加载了 {len(self.data_list)} 个样本")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        Returns:
            video_frames: 视频帧，形状为 (sequence_length, 3, H, W)
            audio_features: 音频特征，形状为 (sequence_length, audio_dim)
            label: 标签（1表示匹配，0表示不匹配）
        """
        # 加载数据
        with open(self.data_list[idx]['data_path'], 'rb') as f:
            data = pickle.load(f)
        
        faces = data['faces']
        audio_features = data['audio_features']
        
        length = min(len(faces), len(audio_features))
        
        # 随机选择起始位置
        if self.is_training and length > self.sequence_length:
            start_idx = random.randint(0, length - self.sequence_length)
        else:
            start_idx = 0
        
        # 提取序列
        video_seq = faces[start_idx:start_idx + self.sequence_length]
        audio_seq = audio_features[start_idx:start_idx + self.sequence_length]
        
        # 转换为tensor
        video_seq = torch.FloatTensor(video_seq).permute(0, 3, 1, 2) / 255.0  # (seq, 3, H, W)
        audio_seq = torch.FloatTensor(audio_seq)  # (seq, audio_dim)
        
        # 如果是训练模式，随机生成正负样本对
        if self.is_training:
            # 50%概率为正样本（匹配），50%为负样本（不匹配）
            if random.random() < 0.5:
                label = 1  # 正样本
            else:
                label = 0  # 负样本
                # 对于负样本，使用随机打乱的音频
                audio_seq = audio_seq[torch.randperm(self.sequence_length)]
        else:
            label = 1  # 验证/测试时使用正样本
        
        return video_seq, audio_seq, label


def collate_fn(batch):
    """自定义批处理函数"""
    video_seqs, audio_seqs, labels = zip(*batch)
    
    # 堆叠为batch
    video_batch = torch.stack(video_seqs)  # (batch, seq, 3, H, W)
    audio_batch = torch.stack(audio_seqs)  # (batch, seq, audio_dim)
    label_batch = torch.LongTensor(labels)  # (batch,)
    
    return video_batch, audio_batch, label_batch


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (video_batch, audio_batch, labels) in enumerate(pbar):
        # 移动到设备
        video_batch = video_batch.to(device)  # (batch, seq, 3, H, W)
        audio_batch = audio_batch.to(device)  # (batch, seq, audio_dim)
        labels = labels.to(device)  # (batch,)
        
        batch_size = video_batch.size(0)
        seq_len = video_batch.size(1)
        
        # 处理每个时间步
        losses = []
        predictions = []
        
        for t in range(seq_len):
            video_frame = video_batch[:, t, :, :, :]  # (batch, 3, H, W)
            audio_feat = audio_batch[:, :t+1, :]  # (batch, t+1, audio_dim)
            
            # 前向传播
            score = model(video_frame, audio_feat)  # (batch, 1)
            predictions.append(score)
            
            # 计算损失
            if labels[0] == 1:  # 正样本
                # 正样本应该得分高
                target = torch.ones_like(score)
                loss = nn.functional.mse_loss(score, target)
            else:  # 负样本
                # 负样本应该得分低
                target = torch.zeros_like(score)
                loss = nn.functional.mse_loss(score, target)
            
            losses.append(loss)
        
        # 平均损失
        total_loss_batch = sum(losses) / len(losses)
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # 统计
        avg_score = torch.mean(torch.cat(predictions, dim=0))
        pred = (avg_score > 0.5).long()
        correct += (pred == labels[0]).item()
        total += 1
        
        total_loss += total_loss_batch.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
        
        # 记录日志
        if batch_idx % config['training']['log_interval'] == 0:
            global_step = epoch * len(dataloader) + batch_idx
            # 这里可以添加tensorboard日志记录
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for video_batch, audio_batch, labels in tqdm(dataloader, desc="验证"):
            video_batch = video_batch.to(device)
            audio_batch = audio_batch.to(device)
            labels = labels.to(device)
            
            batch_size = video_batch.size(0)
            seq_len = video_batch.size(1)
            
            predictions = []
            
            for t in range(seq_len):
                video_frame = video_batch[:, t, :, :, :]
                audio_feat = audio_batch[:, :t+1, :]
                
                score = model(video_frame, audio_feat)
                predictions.append(score)
            
            avg_score = torch.mean(torch.cat(predictions, dim=0))
            pred = (avg_score > 0.5).long()
            correct += (pred == labels[0]).item()
            total += 1
            
            if labels[0] == 1:
                target = torch.ones_like(avg_score)
            else:
                target = torch.zeros_like(avg_score)
            
            loss = nn.functional.mse_loss(avg_score, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    """主训练函数"""
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(config['paths']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据集
    data_index_path = Path(config['data']['output_dir']) / "dataset_index.pkl"
    
    if not data_index_path.exists():
        print("错误: 未找到处理后的数据，请先运行数据预处理！")
        print("运行命令: python src/data_processing.py")
        return
    
    # 划分数据集
    dataset = LipSyncDataset(
        data_index_path, 
        sequence_length=config['data']['sequence_length'],
        is_training=True
    )
    
    total_size = len(dataset)
    train_size = int(total_size * config['data']['train_split'])
    val_size = int(total_size * config['data']['val_split'])
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows上可能需要设为0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练循环
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}, 测试样本: {len(test_dataset)}")
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"保存最佳模型: {checkpoint_path}")
        
        # 定期保存检查点
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
    
    writer.close()
    print("\n训练完成！")


if __name__ == "__main__":
    main()

