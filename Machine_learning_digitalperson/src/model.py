"""
模型定义模块
定义用于口型-音频匹配的深度学习模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoEncoder(nn.Module):
    """视频编码器，用于提取视频帧特征"""
    
    def __init__(self, input_size=96, hidden_dim=512):
        """
        初始化视频编码器
        Args:
            input_size: 输入图像尺寸
            hidden_dim: 隐藏层维度
        """
        super(VideoEncoder, self).__init__()
        
        # CNN特征提取
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 计算特征图尺寸
        feature_size = input_size // 16  # 经过4次stride=2的卷积
        self.fc = nn.Linear(512 * feature_size * feature_size, hidden_dim)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像，形状为 (batch, 3, H, W)
        Returns:
            features: 特征向量，形状为 (batch, hidden_dim)
        """
        # CNN特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # 展平
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = self.fc(x)
        
        return x


class AudioEncoder(nn.Module):
    """音频编码器，用于提取音频特征"""
    
    def __init__(self, input_dim=13, hidden_dim=512, num_layers=2):
        """
        初始化音频编码器
        Args:
            input_dim: 输入特征维度（MFCC维度）
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
        """
        super(AudioEncoder, self).__init__()
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        # 将双向LSTM输出映射到hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入音频特征，形状为 (batch, seq_len, input_dim)
        Returns:
            features: 特征向量，形状为 (batch, hidden_dim)
        """
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        # lstm_out形状: (batch, seq_len, hidden_dim * 2)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        # 全连接层
        features = self.fc(last_output)
        
        return features


class LipSyncModel(nn.Module):
    """口型-音频匹配模型"""
    
    def __init__(self, config):
        """
        初始化模型
        Args:
            config: 配置字典
        """
        super(LipSyncModel, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        # 视频编码器
        self.video_encoder = VideoEncoder(
            input_size=model_config['input_size'],
            hidden_dim=model_config['hidden_dim']
        )
        
        # 音频编码器
        self.audio_encoder = AudioEncoder(
            input_dim=model_config['audio_feature_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config.get('num_layers', 2)
        )
        
        # 融合层
        hidden_dim = model_config['hidden_dim']
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.get('dropout', 0.3)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出匹配分数
        )
        
    def forward(self, video_frames, audio_features):
        """
        前向传播
        Args:
            video_frames: 视频帧，形状为 (batch, 3, H, W)
            audio_features: 音频特征，形状为 (batch, seq_len, audio_dim)
        Returns:
            score: 匹配分数，形状为 (batch, 1)
        """
        # 编码视频
        video_feat = self.video_encoder(video_frames)  # (batch, hidden_dim)
        
        # 编码音频
        audio_feat = self.audio_encoder(audio_features)  # (batch, hidden_dim)
        
        # 融合特征
        combined = torch.cat([video_feat, audio_feat], dim=1)  # (batch, hidden_dim * 2)
        
        # 计算匹配分数
        score = self.fusion(combined)  # (batch, 1)
        
        return score
    
    def encode_video(self, video_frames):
        """仅编码视频"""
        return self.video_encoder(video_frames)
    
    def encode_audio(self, audio_features):
        """仅编码音频"""
        return self.audio_encoder(audio_features)


class ContrastiveLoss(nn.Module):
    """对比损失函数，用于训练口型-音频匹配"""
    
    def __init__(self, margin=1.0):
        """
        初始化对比损失
        Args:
            margin: 边界值
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, positive_scores, negative_scores):
        """
        计算对比损失
        Args:
            positive_scores: 正样本（匹配）的分数，形状为 (batch, 1)
            negative_scores: 负样本（不匹配）的分数，形状为 (batch, 1)
        Returns:
            loss: 损失值
        """
        # 正样本应该得分高，负样本应该得分低
        # 使用margin ranking loss的思想
        loss = torch.mean(
            torch.clamp(
                self.margin - positive_scores + negative_scores,
                min=0.0
            )
        )
        
        return loss


def create_model(config):
    """
    创建模型实例
    Args:
        config: 配置字典
    Returns:
        model: 模型实例
    """
    model = LipSyncModel(config)
    return model


if __name__ == "__main__":
    # 测试模型
    import yaml
    
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # 创建虚拟输入
    batch_size = 4
    video_input = torch.randn(batch_size, 3, 96, 96)
    audio_input = torch.randn(batch_size, 5, 13)  # seq_len=5, audio_dim=13
    
    # 前向传播
    output = model(video_input, audio_input)
    print(f"模型输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

