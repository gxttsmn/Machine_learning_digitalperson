"""
推理接口模块
提供模型推理和调用接口
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import librosa
import yaml
from pathlib import Path
import argparse
from model import create_model
from data_processing import VideoProcessor, AudioProcessor


class LipSyncInference:
    """口型-音频匹配推理类"""
    
    def __init__(self, checkpoint_path, config_path="config.yaml"):
        """
        初始化推理器
        Args:
            checkpoint_path: 模型检查点路径
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device(
            self.config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # 创建模型
        self.model = create_model(self.config)
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功: {checkpoint_path}")
        print(f"使用设备: {self.device}")
        
        # 初始化处理器
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['data']['audio_sample_rate'],
            n_mfcc=self.config['model']['audio_feature_dim']
        )
    
    def compute_sync_score(self, video_path, audio_path):
        """
        计算视频和音频的同步分数
        Args:
            video_path: 视频文件路径
            audio_path: 音频文件路径
        Returns:
            score: 同步分数 (0-1之间，越高表示越同步)
            details: 详细信息字典
        """
        # 提取视频帧
        faces, timestamps = self.video_processor.extract_faces(
            video_path,
            output_size=self.config['model']['input_size'],
            fps=self.config['data']['fps']
        )
        
        if len(faces) == 0:
            return 0.0, {"error": "未检测到人脸"}
        
        # 提取音频特征
        audio_features = self.audio_processor.extract_features(
            audio_path,
            timestamps
        )
        
        # 确保长度一致
        min_len = min(len(faces), len(audio_features))
        faces = faces[:min_len]
        audio_features = audio_features[:min_len]
        
        # 转换为tensor
        video_tensor = torch.FloatTensor(faces).permute(0, 3, 1, 2) / 255.0
        audio_tensor = torch.FloatTensor(audio_features)
        
        # 移动到设备
        video_tensor = video_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        
        # 计算每个时间步的分数
        scores = []
        
        with torch.no_grad():
            for t in range(min_len):
                video_frame = video_tensor[t:t+1]  # (1, 3, H, W)
                audio_feat = audio_tensor[:t+1].unsqueeze(0)  # (1, t+1, audio_dim)
                
                score = self.model(video_frame, audio_feat)
                scores.append(score.item())
        
        # 平均分数
        avg_score = np.mean(scores)
        # 归一化到0-1
        normalized_score = max(0.0, min(1.0, (avg_score + 1) / 2))
        
        details = {
            "raw_score": avg_score,
            "normalized_score": normalized_score,
            "frame_scores": scores,
            "num_frames": min_len
        }
        
        return normalized_score, details
    
    def sync_video_audio(self, video_path, audio_path, output_path, threshold=0.5):
        """
        同步视频和音频，生成输出视频
        Args:
            video_path: 输入视频路径
            audio_path: 输入音频路径
            output_path: 输出视频路径
            threshold: 同步阈值，低于此值会给出警告
        Returns:
            success: 是否成功
            message: 消息
        """
        # 计算同步分数
        score, details = self.compute_sync_score(video_path, audio_path)
        
        if score < threshold:
            print(f"警告: 同步分数较低 ({score:.3f})，可能不匹配")
        
        # 读取原始视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 读取音频
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 处理视频帧（这里只是简单复制，实际应用中可能需要更复杂的处理）
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 可以在这里添加基于模型输出的后处理
            # 例如：根据同步分数调整帧
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        # 注意：这里没有实际替换音频，实际应用中需要使用ffmpeg等工具
        print(f"视频已保存: {output_path}")
        print(f"同步分数: {score:.3f}")
        
        return True, f"处理完成，同步分数: {score:.3f}"
    
    def batch_process(self, video_audio_pairs, output_dir):
        """
        批量处理视频-音频对
        Args:
            video_audio_pairs: 列表，每个元素为(video_path, audio_path)
            output_dir: 输出目录
        Returns:
            results: 处理结果列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, (video_path, audio_path) in enumerate(video_audio_pairs):
            print(f"\n处理 {i+1}/{len(video_audio_pairs)}: {Path(video_path).name}")
            
            try:
                score, details = self.compute_sync_score(video_path, audio_path)
                
                output_path = output_dir / f"output_{i:04d}.mp4"
                success, message = self.sync_video_audio(
                    video_path, audio_path, str(output_path)
                )
                
                results.append({
                    "video": video_path,
                    "audio": audio_path,
                    "score": score,
                    "success": success,
                    "output": str(output_path),
                    "message": message
                })
                
            except Exception as e:
                print(f"处理失败: {e}")
                results.append({
                    "video": video_path,
                    "audio": audio_path,
                    "score": 0.0,
                    "success": False,
                    "error": str(e)
                })
        
        return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='口型-音频匹配推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--video', type=str,
                       help='输入视频路径')
    parser.add_argument('--audio', type=str,
                       help='输入音频路径')
    parser.add_argument('--output', type=str,
                       help='输出视频路径')
    parser.add_argument('--score-only', action='store_true',
                       help='仅计算同步分数，不生成输出视频')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = LipSyncInference(args.checkpoint, args.config)
    
    if args.video and args.audio:
        if args.score_only:
            # 仅计算分数
            score, details = inference.compute_sync_score(args.video, args.audio)
            print(f"\n同步分数: {score:.4f}")
            print(f"详细信息: {details}")
        else:
            # 生成输出视频
            if not args.output:
                args.output = "output_synced.mp4"
            
            success, message = inference.sync_video_audio(
                args.video, args.audio, args.output
            )
            print(f"\n{message}")
    else:
        print("请提供视频和音频路径！")
        print("使用 --help 查看帮助信息")


if __name__ == "__main__":
    main()

