"""
数据预处理模块
用于处理视频和音频数据，提取特征用于模型训练
"""

import os
import cv2
import numpy as np
import librosa
import yaml
from pathlib import Path
from tqdm import tqdm
import face_alignment
from scipy.spatial.distance import cdist
import pickle
import soundfile as sf


class VideoProcessor:
    """视频处理类，用于提取视频帧和面部区域"""
    
    def __init__(self, face_detector=None):
        """
        初始化视频处理器
        Args:
            face_detector: 人脸检测器，如果为None则使用face_alignment
        """
        if face_detector is None:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False, 
                device='cpu'
            )
        else:
            self.fa = face_detector
    
    def extract_faces(self, video_path, output_size=96, fps=25):
        """
        从视频中提取人脸区域
        Args:
            video_path: 视频文件路径
            output_size: 输出图像尺寸
            fps: 目标帧率
        Returns:
            faces: 提取的人脸图像列表
            timestamps: 对应的时间戳列表
        """
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(original_fps / fps))
        
        faces = []
        timestamps = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 检测人脸关键点
                try:
                    landmarks = self.fa.get_landmarks(frame_rgb)
                    if landmarks is not None and len(landmarks) > 0:
                        # 获取人脸边界框
                        face_bbox = self._get_face_bbox(landmarks[0], frame_rgb.shape)
                        
                        # 裁剪并调整大小
                        face = self._crop_face(frame_rgb, face_bbox, output_size)
                        faces.append(face)
                        timestamps.append(frame_idx / original_fps)
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue
            
            frame_idx += 1
        
        cap.release()
        return np.array(faces), np.array(timestamps)
    
    def _get_face_bbox(self, landmarks, img_shape):
        """根据关键点计算人脸边界框"""
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = max(0, int(np.min(x_coords) * 0.9))
        x_max = min(img_shape[1], int(np.max(x_coords) * 1.1))
        y_min = max(0, int(np.min(y_coords) * 0.9))
        y_max = min(img_shape[0], int(np.max(y_coords) * 1.1))
        
        return (x_min, y_min, x_max, y_max)
    
    def _crop_face(self, frame, bbox, output_size):
        """裁剪并调整人脸区域大小"""
        x_min, y_min, x_max, y_max = bbox
        
        # 确保是正方形
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - size // 2)
        y_min = max(0, center_y - size // 2)
        x_max = min(frame.shape[1], x_min + size)
        y_max = min(frame.shape[0], y_min + size)
        
        face = frame[y_min:y_max, x_min:x_max]
        face = cv2.resize(face, (output_size, output_size))
        
        return face


class AudioProcessor:
    """音频处理类，用于提取音频特征"""
    
    def __init__(self, sample_rate=16000, n_mfcc=13):
        """
        初始化音频处理器
        Args:
            sample_rate: 采样率
            n_mfcc: MFCC特征数量
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path, timestamps, hop_length=512):
        """
        从音频中提取MFCC特征
        Args:
            audio_path: 音频文件路径
            timestamps: 视频帧的时间戳
            hop_length: 帧移长度
        Returns:
            features: MFCC特征数组，形状为 (n_frames, n_mfcc)
        """
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.n_mfcc,
            hop_length=hop_length
        )
        mfccs = mfccs.T  # 转置为 (time, n_mfcc)
        
        # 根据时间戳对齐特征
        frame_times = librosa.frames_to_time(
            np.arange(mfccs.shape[0]), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # 为每个视频帧找到对应的音频特征
        aligned_features = []
        for ts in timestamps:
            # 找到最接近的时间戳
            idx = np.argmin(np.abs(frame_times - ts))
            aligned_features.append(mfccs[idx])
        
        return np.array(aligned_features)


class DataPreprocessor:
    """数据预处理器主类"""
    
    def __init__(self, config_path="config.yaml"):
        """
        初始化数据预处理器
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['data']['audio_sample_rate'],
            n_mfcc=self.config['model']['audio_feature_dim']
        )
        
        self.video_dir = Path(self.config['data']['video_dir'])
        self.audio_dir = Path(self.config['data']['audio_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self):
        """处理整个数据集"""
        video_files = list(self.video_dir.glob("*.mp4")) + \
                     list(self.video_dir.glob("*.avi"))
        
        processed_data = []
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        for video_path in tqdm(video_files, desc="处理视频"):
            # 查找对应的音频文件
            audio_path = self.audio_dir / (video_path.stem + ".wav")
            if not audio_path.exists():
                audio_path = self.audio_dir / (video_path.stem + ".mp3")
            
            if not audio_path.exists():
                print(f"警告: 未找到 {video_path.stem} 对应的音频文件")
                continue
            
            try:
                # 提取视频帧
                faces, timestamps = self.video_processor.extract_faces(
                    str(video_path),
                    output_size=self.config['model']['input_size'],
                    fps=self.config['data']['fps']
                )
                
                if len(faces) == 0:
                    print(f"警告: {video_path.stem} 未检测到人脸")
                    continue
                
                # 提取音频特征
                audio_features = self.audio_processor.extract_features(
                    str(audio_path),
                    timestamps
                )
                
                # 确保长度一致
                min_len = min(len(faces), len(audio_features))
                faces = faces[:min_len]
                audio_features = audio_features[:min_len]
                
                # 保存处理后的数据
                data_item = {
                    'faces': faces,
                    'audio_features': audio_features,
                    'video_path': str(video_path),
                    'audio_path': str(audio_path)
                }
                
                output_path = self.output_dir / f"{video_path.stem}.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(data_item, f)
                
                processed_data.append({
                    'data_path': str(output_path),
                    'length': min_len
                })
                
            except Exception as e:
                print(f"处理 {video_path.stem} 时出错: {e}")
                continue
        
        # 保存数据集索引
        index_path = self.output_dir / "dataset_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\n处理完成！共处理 {len(processed_data)} 个样本")
        print(f"数据保存在: {self.output_dir}")
        print(f"索引文件: {index_path}")
        
        return processed_data


if __name__ == "__main__":
    # 运行数据预处理
    preprocessor = DataPreprocessor()
    preprocessor.process_dataset()

