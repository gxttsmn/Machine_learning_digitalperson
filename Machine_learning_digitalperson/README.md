# 视频人像口型与语音匹配模型训练项目

本项目用于训练一个本地可调用的垂类领域大模型，主要应用于视频人像口型与语音匹配功能（Lip-Sync）。

## 项目结构

```
Machine_learning/
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包
├── README.md               # 项目说明
├── data/                   # 数据目录
│   ├── videos/            # 原始视频文件
│   ├── audios/            # 原始音频文件
│   └── processed/         # 处理后的数据
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── data_processing.py # 数据预处理模块
│   ├── model.py           # 模型定义
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理接口
├── checkpoints/            # 模型检查点
├── logs/                   # 训练日志
└── outputs/                # 输出结果
```

## 详细操作流程

**快速开始请查看 [QUICK_START.md](QUICK_START.md) 获取详细的分步指南！**

### 第一步：环境准备

1. **安装Python环境**
   - 推荐使用Python 3.8-3.10
   - 创建虚拟环境：
     ```bash
     python -m venv venv
     venv\Scripts\activate  # Windows
     # 或 source venv/bin/activate  # Linux/Mac
     ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装PyTorch（根据CUDA版本）**
   - 如果有NVIDIA GPU，访问 https://pytorch.org/ 获取对应CUDA版本的安装命令
   - 例如：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

4. **创建项目目录**
   ```bash
   python setup_directories.py
   ```

### 第二步：数据准备

1. **准备训练数据**
   - 将视频文件放入 `data/videos/` 目录
   - 将对应的音频文件放入 `data/audios/` 目录
   - 确保视频和音频文件名对应（例如：video1.mp4 对应 video1.wav）

2. **数据要求**
   - 视频格式：MP4, AVI等常见格式
   - 音频格式：WAV, MP3等
   - 视频应包含清晰的人脸正面图像
   - 音频应与视频同步
   - 建议每个视频长度在5-30秒之间
   - **最小数据量：建议至少100个视频-音频对**

3. **运行数据预处理**
   ```bash
   python src/data_processing.py
   ```
   这会自动：
   - 检测视频中的人脸
   - 提取人脸区域
   - 提取音频MFCC特征
   - 对齐视频帧和音频特征
   - 保存处理后的数据

### 第三步：模型训练

1. **检查配置文件**
   - 编辑 `config.yaml` 调整训练参数：
     - `batch_size`: 根据GPU内存调整（6GB显存建议8-16）
     - `learning_rate`: 学习率（默认0.0001）
     - `num_epochs`: 训练轮数（默认100）

2. **开始训练**
   ```bash
   python src/train.py
   ```
   
   训练过程会：
   - 自动划分训练集/验证集/测试集（8:1:1）
   - 显示实时训练进度和损失
   - 自动保存最佳模型到 `checkpoints/best_model.pth`
   - 每10个epoch保存检查点

3. **监控训练过程（可选）**
   - 使用TensorBoard查看训练曲线：
     ```bash
     tensorboard --logdir logs
     ```
   - 在浏览器打开 http://localhost:6006

### 第四步：模型推理

1. **Python代码调用**
   ```python
   from src.inference import LipSyncInference
   
   # 初始化推理器
   inference = LipSyncInference(checkpoint_path="checkpoints/best_model.pth")
   
   # 计算同步分数
   score, details = inference.compute_sync_score(
       video_path="test_video.mp4",
       audio_path="test_audio.wav"
   )
   print(f"同步分数: {score:.3f}")  # 0-1之间，越高越同步
   
   # 生成同步视频
   success, message = inference.sync_video_audio(
       video_path="input_video.mp4",
       audio_path="input_audio.wav",
       output_path="output_video.mp4"
   )
   ```

2. **命令行调用**
   ```bash
   # 仅计算同步分数
   python src/inference.py \
       --checkpoint checkpoints/best_model.pth \
       --video input_video.mp4 \
       --audio input_audio.wav \
       --score-only
   
   # 生成同步视频
   python src/inference.py \
       --checkpoint checkpoints/best_model.pth \
       --video input_video.mp4 \
       --audio input_audio.wav \
       --output output_video.mp4
   ```

3. **查看示例代码**
   - 查看 `example_usage.py` 了解更多使用示例

## 技术说明

### 模型架构
- 使用CNN提取视频帧特征
- 使用MFCC提取音频特征
- 使用LSTM/Transformer进行时序建模
- 输出口型同步的置信度分数

### 训练策略
- 使用对比学习训练口型-音频匹配
- 数据增强：随机裁剪、翻转、噪声添加
- 学习率调度：余弦退火

## 注意事项

1. **硬件要求**
   - 推荐使用NVIDIA GPU（至少6GB显存）
   - CPU训练会非常慢

2. **数据质量**
   - 高质量的训练数据是模型效果的关键
   - 建议至少准备1000个视频-音频对

3. **训练时间**
   - 根据数据量和硬件配置，训练可能需要数小时到数天

## 常见问题

1. **CUDA out of memory**
   - 减小batch_size
   - 减小输入图像尺寸

2. **数据加载错误**
   - 检查视频和音频文件格式
   - 确保文件路径正确

3. **模型不收敛**
   - 调整学习率
   - 检查数据质量
   - 增加训练数据量

## 后续优化方向

- 支持实时推理
- 改进模型架构（使用Transformer）
- 支持多人脸检测
- 提高处理速度

