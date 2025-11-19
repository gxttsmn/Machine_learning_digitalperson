"""
示例使用脚本
演示如何使用训练好的模型进行推理
"""

from src.inference import LipSyncInference
from pathlib import Path

def example_compute_score():
    """示例：计算视频和音频的同步分数"""
    print("=" * 50)
    print("示例1: 计算同步分数")
    print("=" * 50)
    
    # 初始化推理器
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        print("请先训练模型或使用已训练的模型")
        return
    
    inference = LipSyncInference(checkpoint_path)
    
    # 计算同步分数
    video_path = "test_video.mp4"  # 替换为你的视频路径
    audio_path = "test_audio.wav"  # 替换为你的音频路径
    
    if not Path(video_path).exists() or not Path(audio_path).exists():
        print(f"请确保视频和音频文件存在:")
        print(f"  视频: {video_path}")
        print(f"  音频: {audio_path}")
        return
    
    score, details = inference.compute_sync_score(video_path, audio_path)
    
    print(f"\n同步分数: {score:.4f} (0-1之间，越高越同步)")
    print(f"详细信息:")
    print(f"  - 原始分数: {details.get('raw_score', 'N/A'):.4f}")
    print(f"  - 处理帧数: {details.get('num_frames', 'N/A')}")
    
    if score > 0.7:
        print("\n✓ 视频和音频高度同步")
    elif score > 0.5:
        print("\n○ 视频和音频基本同步")
    else:
        print("\n✗ 视频和音频可能不同步")


def example_sync_video():
    """示例：生成同步视频"""
    print("\n" + "=" * 50)
    print("示例2: 生成同步视频")
    print("=" * 50)
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        return
    
    inference = LipSyncInference(checkpoint_path)
    
    video_path = "input_video.mp4"  # 替换为你的视频路径
    audio_path = "input_audio.wav"  # 替换为你的音频路径
    output_path = "output_synced.mp4"
    
    if not Path(video_path).exists() or not Path(audio_path).exists():
        print(f"请确保视频和音频文件存在:")
        print(f"  视频: {video_path}")
        print(f"  音频: {audio_path}")
        return
    
    success, message = inference.sync_video_audio(
        video_path, audio_path, output_path
    )
    
    if success:
        print(f"\n✓ {message}")
        print(f"输出视频已保存: {output_path}")
    else:
        print(f"\n✗ 处理失败: {message}")


def example_batch_process():
    """示例：批量处理"""
    print("\n" + "=" * 50)
    print("示例3: 批量处理")
    print("=" * 50)
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not Path(checkpoint_path).exists():
        print(f"错误: 未找到模型文件 {checkpoint_path}")
        return
    
    inference = LipSyncInference(checkpoint_path)
    
    # 准备视频-音频对列表
    video_audio_pairs = [
        ("video1.mp4", "audio1.wav"),
        ("video2.mp4", "audio2.wav"),
        # 添加更多对...
    ]
    
    # 过滤存在的文件
    existing_pairs = [
        (v, a) for v, a in video_audio_pairs 
        if Path(v).exists() and Path(a).exists()
    ]
    
    if not existing_pairs:
        print("未找到有效的视频-音频对")
        return
    
    # 批量处理
    results = inference.batch_process(existing_pairs, "outputs/batch")
    
    # 显示结果
    print(f"\n处理完成，共处理 {len(results)} 个样本:")
    for i, result in enumerate(results):
        status = "✓" if result['success'] else "✗"
        print(f"{status} {i+1}. {Path(result['video']).name}: "
              f"分数={result['score']:.3f}")


if __name__ == "__main__":
    print("\n口型-音频匹配模型使用示例\n")
    print("注意: 请先确保:")
    print("1. 已训练模型 (checkpoints/best_model.pth)")
    print("2. 准备好测试视频和音频文件")
    print("\n")
    
    # 运行示例（取消注释以运行）
    # example_compute_score()
    # example_sync_video()
    # example_batch_process()
    
    print("\n提示: 取消注释 example_usage.py 中的函数调用来运行示例")

