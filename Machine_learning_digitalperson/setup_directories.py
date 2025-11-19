"""
创建项目目录结构
"""

from pathlib import Path

# 需要创建的目录
directories = [
    "data/videos",
    "data/audios",
    "data/processed",
    "checkpoints",
    "logs",
    "outputs"
]

for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {dir_path}")

print("\n目录结构创建完成！")
print("\n下一步:")
print("1. 将视频文件放入 data/videos/ 目录")
print("2. 将音频文件放入 data/audios/ 目录")
print("3. 运行数据预处理: python src/data_processing.py")
print("4. 开始训练: python src/train.py")

