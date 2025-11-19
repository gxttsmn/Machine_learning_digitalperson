# 快速开始指南

## 第一步：环境准备（5-10分钟）

### 1.1 检查Python版本
```bash
python --version
```
确保Python版本在3.8-3.10之间

### 1.2 创建虚拟环境（推荐）
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 1.3 安装依赖
```bash
pip install -r requirements.txt
```

### 1.4 安装PyTorch（根据你的系统）
访问 https://pytorch.org/ 获取适合你系统的安装命令

**有NVIDIA GPU的用户：**
```bash
# CUDA 11.8版本示例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**仅CPU用户：**
```bash
pip install torch torchvision torchaudio
```

### 1.5 创建项目目录
```bash
python setup_directories.py
```

## 第二步：数据准备（30分钟-数小时，取决于数据量）

### 2.1 准备训练数据

1. **收集视频文件**
   - 将视频文件放入 `data/videos/` 目录
   - 支持的格式：MP4, AVI等
   - 要求：
     - 包含清晰的人脸正面图像
     - 视频长度建议5-30秒
     - 分辨率至少320x240

2. **准备音频文件**
   - 将对应的音频文件放入 `data/audios/` 目录
   - 支持的格式：WAV, MP3等
   - 要求：
     - 音频应与视频同步
     - 采样率建议16kHz或更高
     - 单声道或立体声均可

3. **文件命名**
   - 视频和音频文件名应对应
   - 例如：`video1.mp4` 对应 `video1.wav`
   - 或：`person1_talk.mp4` 对应 `person1_talk.wav`

### 2.2 运行数据预处理

```bash
python src/data_processing.py
```

**处理过程：**
- 自动检测视频中的人脸
- 提取人脸区域并调整大小
- 提取音频MFCC特征
- 对齐视频帧和音频特征
- 保存处理后的数据到 `data/processed/`

**预期输出：**
```
找到 100 个视频文件
处理视频: 100%|████████████| 100/100 [05:23<00:00,  3.23s/it]

处理完成！共处理 95 个样本
数据保存在: data/processed
索引文件: data/processed/dataset_index.pkl
```

**常见问题：**
- 如果提示"未检测到人脸"，检查视频中是否有人脸
- 如果提示"未找到对应的音频文件"，检查文件名是否匹配

## 第三步：模型训练（数小时-数天，取决于数据量和硬件）

### 3.1 检查配置文件

编辑 `config.yaml` 调整训练参数：

```yaml
training:
  batch_size: 16        # 根据GPU内存调整（6GB显存建议8-16）
  num_epochs: 100       # 训练轮数
  learning_rate: 0.0001 # 学习率
  device: "cuda"        # 使用GPU，CPU训练会很慢
```

### 3.2 开始训练

```bash
python src/train.py
```

**训练过程：**
- 自动划分训练集/验证集/测试集（默认8:1:1）
- 显示训练进度和损失
- 自动保存最佳模型到 `checkpoints/best_model.pth`
- 每10个epoch保存检查点

**预期输出：**
```
使用设备: cuda
模型参数数量: 2,345,678
加载了 95 个样本

开始训练...
训练样本: 76, 验证样本: 9, 测试样本: 10

Epoch 1/100
训练: 100%|████████████| 5/5 [01:23<00:00, 16.6s/it, loss=0.2341, acc=0.6500]
验证: 100%|████████████| 1/1 [00:05<00:00,  5.2s/it]
训练损失: 0.2341, 训练准确率: 0.6500
验证损失: 0.1892, 验证准确率: 0.7778
学习率: 0.000100
保存最佳模型: checkpoints/best_model.pth
```

### 3.3 监控训练（可选）

在另一个终端运行：
```bash
tensorboard --logdir logs
```

然后在浏览器打开 http://localhost:6006 查看训练曲线

### 3.4 训练建议

- **最小数据量**：建议至少100个视频-音频对
- **训练时间**：
  - GPU（RTX 3060）：约2-4小时/100个epoch
  - CPU：可能需要数天
- **停止训练**：按Ctrl+C可以安全停止，已保存的模型可以使用

## 第四步：模型推理（几分钟）

### 4.1 Python代码调用

```python
from src.inference import LipSyncInference

# 初始化推理器
inference = LipSyncInference(
    checkpoint_path="checkpoints/best_model.pth"
)

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
    output_path="output_synced.mp4"
)
```

### 4.2 命令行调用

```bash
# 仅计算同步分数
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video test_video.mp4 \
    --audio test_audio.wav \
    --score-only

# 生成同步视频
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video input_video.mp4 \
    --audio input_audio.wav \
    --output output_synced.mp4
```

## 常见问题排查

### 问题1：CUDA out of memory
**解决方案：**
- 减小 `batch_size`（在config.yaml中）
- 减小 `input_size`（在config.yaml中）
- 使用CPU训练（会很慢）

### 问题2：训练损失不下降
**可能原因：**
- 学习率太大或太小
- 数据质量不好
- 数据量太少

**解决方案：**
- 调整学习率（尝试0.0001, 0.001, 0.00001）
- 检查数据质量
- 增加训练数据量

### 问题3：检测不到人脸
**解决方案：**
- 确保视频中有清晰的人脸
- 人脸应该正面朝向
- 光线充足
- 尝试不同的视频

### 问题4：模型效果不好
**改进方向：**
- 增加训练数据量（至少1000个样本）
- 提高数据质量
- 调整模型参数（hidden_dim, num_layers等）
- 增加训练轮数

## 下一步优化

1. **提高模型性能**
   - 使用更大的模型（增加hidden_dim）
   - 使用Transformer架构
   - 数据增强

2. **实时推理**
   - 优化模型推理速度
   - 使用TensorRT加速

3. **功能扩展**
   - 支持多人脸检测
   - 支持实时视频流
   - 支持批量处理

## 获取帮助

如果遇到问题：
1. 检查错误信息
2. 查看README.md中的详细说明
3. 检查配置文件是否正确
4. 确保所有依赖已正确安装

