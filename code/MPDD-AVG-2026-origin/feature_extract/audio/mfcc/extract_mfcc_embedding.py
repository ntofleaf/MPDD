# -*- coding: utf-8 -*-
"""
MFCC 音频特征提取 - 原始完整音频版本（无滑窗切分）
处理原始数据目录下的完整音频文件

使用 librosa 提取 MFCC 特征：
- 输入：音频文件
- 输出：64维特征向量（对所有帧特征取平均，utterance-level）
"""
import os
import glob
import numpy as np
import librosa
import argparse
from tqdm import tqdm


def write_feature_to_npy(feature, npy_file, feature_level='FRAME'):
    """
    将特征写入 npy 文件。

    Args:
        feature (numpy.ndarray): 特征数据。
        npy_file (str): 保存特征的文件路径。
        feature_level (str): 特征层次，'UTTERANCE' 或 'FRAME'。
    
    Returns:
        bool: 是否成功保存。
    """
    if feature_level == 'UTTERANCE':
        # Utterance-level: 对时间维度取平均
        feature = np.array(feature).squeeze()  # [C,]
        if len(feature.shape) != 1:  # 如果是 [T, C]，转为 [C,]
            feature = np.mean(feature, axis=0)
    else:
        # Frame-level: 保留时间维度
        feature = np.array(feature).squeeze()
        if len(feature.shape) == 1:  # 如果只有一帧，reshape 为 [1, C]
            feature = feature[np.newaxis, :]
    
    try:
        np.save(npy_file, feature)
        return True
    except Exception as e:
        print(f"  保存特征失败: {e}")
        return False


def extract_mfcc_features(audio_file, feature_level='FRAME', n_mfcc=64, 
                         frame_length=2048, hop_length=512):
    """
    从音频文件中提取 MFCC 特征。

    Args:
        audio_file (str): 音频文件路径。
        feature_level (str): 特征层次，'UTTERANCE' 或 'FRAME'。
        n_mfcc (int): MFCC 特征维度，默认为 64。
        frame_length (int): 帧长度。
        hop_length (int): 跳跃长度。
    
    Returns:
        numpy.ndarray: 提取的特征。
    """
    try:
        # 加载音频文件
        y, sr = librosa.load(audio_file, sr=None)
        
        # 提取 MFCC 特征
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=n_mfcc,
            n_fft=frame_length,
            hop_length=hop_length
        )
        
        # mfcc shape: (n_mfcc, T)，转置为 (T, n_mfcc)
        mfcc = mfcc.T
        
        return mfcc
    
    except Exception as e:
        print(f"  提取特征失败: {e}")
        return None


def process_original_audio(root_dir, save_dir, feature_level='FRAME', 
                          overwrite=False, n_mfcc=64, frame_length=2048, hop_length=512):
    """
    处理原始音频文件（无滑窗切分），提取其特征并保存。
    
    目录结构：
    root_dir/
    ├── 1/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   ...
    ├── 2/
    │   ├── audio1.wav
    │   ...

    Args:
        root_dir (str): 原始音频根目录，包含每个 ID 的文件夹。
        save_dir (str): 保存提取特征的目录。
        feature_level (str): 特征层次，'UTTERANCE' 或 'FRAME'。
        overwrite (bool): 是否覆盖已存在的特征文件。
        n_mfcc (int): MFCC 特征维度，默认为 64。
        frame_length (int): 帧长度。
        hop_length (int): 跳跃长度。
    """
    print('='*60)
    print('MFCC 音频特征提取 - 原始完整音频（无滑窗）')
    print('='*60)
    print(f'特征级别: {feature_level}')
    print(f'MFCC 维度: {n_mfcc}')
    print(f'帧长度: {frame_length}')
    print(f'跳跃长度: {hop_length}')
    print('='*60)

    # 创建输出目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_audios = 0
    processed_audios = 0
    skipped_audios = 0
    failed_audios = 0

    # 遍历根目录下的所有 ID 文件夹
    for id_folder in sorted(os.listdir(root_dir)):
        id_folder_path = os.path.join(root_dir, id_folder)
        if not os.path.isdir(id_folder_path):
            continue

        print(f'\n正在处理 ID 文件夹: {id_folder}')

        # 创建对应的输出文件夹
        id_output_folder = os.path.join(save_dir, id_folder)
        if not os.path.exists(id_output_folder):
            os.makedirs(id_output_folder)

        # 查找音频文件
        audio_files = []
        audio_files += glob.glob(os.path.join(id_folder_path, '*.wav'))
        audio_files += glob.glob(os.path.join(id_folder_path, '*.mp3'))
        audio_files += glob.glob(os.path.join(id_folder_path, '*.flac'))
        audio_files += glob.glob(os.path.join(id_folder_path, '*.WAV'))
        
        if not audio_files:
            print('  未找到音频文件')
            continue

        print(f'  找到 {len(audio_files)} 个音频文件')

        # 为每个音频提取特征
        for audio_file in tqdm(audio_files, desc=f"  处理 {id_folder}"):
            audio_name = os.path.basename(audio_file)
            feature_save_name = f"{os.path.splitext(audio_name)[0]}.npy"
            feature_save_path = os.path.join(id_output_folder, feature_save_name)

            total_audios += 1

            # 检查特征文件是否已存在
            if not overwrite and os.path.exists(feature_save_path):
                skipped_audios += 1
                continue

            # 提取特征
            features = extract_mfcc_features(
                audio_file,
                feature_level=feature_level,
                n_mfcc=n_mfcc,
                frame_length=frame_length,
                hop_length=hop_length
            )

            if features is not None:
                # 保存特征
                success = write_feature_to_npy(features, feature_save_path, feature_level)
                if success:
                    processed_audios += 1
                else:
                    failed_audios += 1
            else:
                failed_audios += 1

    print(f'\n{"="*60}')
    print('处理完成')
    print(f'   总音频数: {total_audios}')
    print(f'   已处理: {processed_audios}')
    print(f'   已跳过: {skipped_audios}')
    print(f'   失败: {failed_audios}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract MFCC features from original audio files (no sliding window).')
    parser.add_argument('--root_dir', type=str, 
                        default='/home/disk2/zelin/MM2025_Challenge/withdisease/basicdisease/原始数据',
                        help='Root directory containing audio files')
    parser.add_argument('--save_dir', type=str, 
                        default='/home/disk2/shangming/feature_extract/feature_frame/audio/mfcc',
                        help='Directory to save extracted features')
    parser.add_argument('--feature_level', type=str, default='FRAME', 
                        choices=['UTTERANCE', 'FRAME'],
                        help='Feature level, either "FRAME" or "UTTERANCE". Default: FRAME')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Whether to overwrite existing feature files.')
    parser.add_argument('--n_mfcc', type=int, default=64,
                        help='Number of MFCC coefficients. Default: 64')
    parser.add_argument('--frame_length', type=int, default=2048,
                        help='Frame length for MFCC extraction. Default: 2048')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='Hop length for MFCC extraction. Default: 512')
    args = parser.parse_args()

    print('='*60)
    print('MFCC 音频特征提取 - 原始完整音频（无滑窗）')
    print('='*60)
    print(f'输入目录: {args.root_dir}')
    print(f'输出目录: {args.save_dir}')
    print(f'特征级别: {args.feature_level}')
    print(f'MFCC 维度: {args.n_mfcc}')
    print(f'覆盖已有文件: {args.overwrite}')
    print('='*60)

    process_original_audio(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        feature_level=args.feature_level,
        overwrite=args.overwrite,
        n_mfcc=args.n_mfcc,
        frame_length=args.frame_length,
        hop_length=args.hop_length
    )
