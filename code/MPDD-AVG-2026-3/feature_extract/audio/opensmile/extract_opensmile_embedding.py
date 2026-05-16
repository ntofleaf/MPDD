# -*- coding: utf-8 -*-
"""
OpenSMILE 音频特征提取 - 原始完整音频版本（无滑窗切分）
处理原始数据目录下的完整音频文件

使用 OpenSMILE 提取音频特征：
- 输入：音频文件
- 输出：特征向量（utterance-level，对所有帧取平均）
- 特征集：ComParE_2016（6373维）或其他
"""
import os
import glob
import numpy as np
import opensmile
import argparse
from tqdm import tqdm


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


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


def extract_opensmile_features(audio_file, smile, feature_level='FRAME'):
    """
    从音频文件中提取 OpenSMILE 特征。

    Args:
        audio_file (str): 音频文件路径。
        smile (opensmile.Smile): OpenSMILE 配置的实例。
        feature_level (str): 特征层次，'UTTERANCE' 或 'FRAME'。
    
    Returns:
        numpy.ndarray: 提取的特征。
    """
    try:
        # 提取特征
        features = smile.process_file(audio_file)
        
        # 转换为 numpy 数组
        features = features.values
        
        return features
    
    except Exception as e:
        print(f"  提取特征失败: {e}")
        return None


def cleanup_generated_image_frames(save_dir):
    """
    删除输出目录中遗留的图片帧文件，仅保留特征 .npy。
    """
    removed_count = 0
    for root, _, files in os.walk(save_dir):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(root, filename)
                try:
                    os.remove(image_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  [WARN] 删除图片帧失败: {image_path}, 错误: {e}")
    return removed_count


def process_original_audio(root_dir, save_dir, feature_level='FRAME', 
                          overwrite=False, feature_set='ComParE_2016', 
                          feature_level_opensmile='LowLevelDescriptors'):
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
        feature_set (str): OpenSMILE 特征集，默认 'ComParE_2016'。
        feature_level_opensmile (str): OpenSMILE 特征级别，'Functionals' 或 'LowLevelDescriptors'。
    """
    print('='*60)
    print('OpenSMILE 音频特征提取 - 原始完整音频（无滑窗）')
    print('='*60)
    print(f'特征级别: {feature_level}')
    print(f'特征集: {feature_set}')
    print(f'OpenSMILE 级别: {feature_level_opensmile}')
    print('='*60)

    # 初始化 OpenSMILE
    try:
        smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, feature_set),
            feature_level=getattr(opensmile.FeatureLevel, feature_level_opensmile),
        )
        print('OpenSMILE 初始化成功')
    except Exception as e:
        print(f'OpenSMILE 初始化失败: {e}')
        return

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
            features = extract_opensmile_features(
                audio_file,
                smile,
                feature_level=feature_level
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

    removed_images = cleanup_generated_image_frames(save_dir)
    print(f'[CLEANUP] 已清理图片帧文件: {removed_images}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract OpenSMILE features from original audio files (no sliding window).')
    parser.add_argument('--root_dir', type=str, 
                        default='/home/disk2/zelin/MM2025_Challenge/withdisease/basicdisease/原始数据',
                        help='Root directory containing audio files')
    parser.add_argument('--save_dir', type=str, 
                        default='/home/disk2/shangming/feature_extract/feature_frame/audio/opensmile',
                        help='Directory to save extracted features')
    parser.add_argument('--feature_level', type=str, default='FRAME', 
                        choices=['UTTERANCE', 'FRAME'],
                        help='Feature level, either "FRAME" or "UTTERANCE". Default: FRAME')
    parser.add_argument('--overwrite', action='store_true', 
                        help='Whether to overwrite existing feature files.')
    parser.add_argument('--feature_set', type=str, default='ComParE_2016',
                        choices=['ComParE_2016', 'GeMAPSv01b', 'eGeMAPSv02'],
                        help='OpenSMILE feature set. Default: ComParE_2016')
    parser.add_argument('--feature_level_opensmile', type=str, default='LowLevelDescriptors',
                        choices=['Functionals', 'LowLevelDescriptors'],
                        help='OpenSMILE feature level. Default: LowLevelDescriptors')
    args = parser.parse_args()

    print('='*60)
    print('OpenSMILE 音频特征提取 - 原始完整音频（无滑窗）')
    print('='*60)
    print(f'输入目录: {args.root_dir}')
    print(f'输出目录: {args.save_dir}')
    print(f'特征级别: {args.feature_level}')
    print(f'特征集: {args.feature_set}')
    print(f'覆盖已有文件: {args.overwrite}')
    print('='*60)

    process_original_audio(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        feature_level=args.feature_level,
        overwrite=args.overwrite,
        feature_set=args.feature_set,
        feature_level_opensmile=args.feature_level_opensmile
    )
