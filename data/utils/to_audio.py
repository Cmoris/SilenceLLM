import subprocess
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from enum import Enum
import re


class SegmentMode(Enum):
    TIME = "time"
    FRAMES = "frames"
    OVERLAP = "overlap"

@dataclass
class VideoSegmentConfig:
    """视频分割配置"""
    window_duration: float = 10.0
    step_duration: float = 5.0
    window_frames: int = 300
    step_frames: int = 150
    overlap_duration: float = 2.0
    copy_codec: bool = True
    prefix: str = "segment"
    max_workers: int = 4
    mode: SegmentMode = SegmentMode.TIME
    output_dir: str = "output"

class AudioFormat(Enum):
    MP3 = "mp3"
    WAV = "wav"
    AAC = "aac"
    FLAC = "flac"
    OGG = "ogg"

@dataclass
class AudioExtractConfig:
    """音频提取配置"""
    audio_format: AudioFormat = AudioFormat.MP3
    quality: str = "high"  # high, medium, low
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    copy_codec: bool = True  # 无损复制音频流
    max_workers: int = 4
    

class FFmpegAudioExtractor:
    """FFmpeg音频提取器"""
    
    def __init__(self, video_path: str):
        """初始化音频提取器"""
        self.video_path = Path(video_path)
        self.lock = threading.Lock()
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        self.video_info = self._get_video_info()
    
    def _get_video_info(self) -> Dict:
        """获取视频基本信息"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(self.video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # 查找音频流
            audio_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
            
            return {
                'duration': float(info['format'].get('duration', 0)),
                'audio_codec': audio_stream.get('codec_name', '') if audio_stream else '',
                'sample_rate': int(audio_stream.get('sample_rate', 0)) if audio_stream else 0,
                'channels': int(audio_stream.get('channels', 0)) if audio_stream else 0,
                'has_audio': audio_stream is not None
            }
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"获取视频信息失败: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"解析视频信息失败: {e}")
    
    def extract_audio(self,
                     output_file: str,
                     config: AudioExtractConfig = None) -> bool:
        """
        提取音频
        
        Args:
            output_file: 输出音频文件路径
            config: 音频提取配置
            
        Returns:
            bool: 是否成功
        """
        if config is None:
            config = AudioExtractConfig()
        
        # 检查是否有音频流
        if not self.video_info['has_audio']:
            print(f"警告: 视频文件 {self.video_path} 没有音频流")
            return False
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.copy_codec:
            # 无损复制音频流（最快）
            return self._extract_audio_copy(output_file, config)
        else:
            # 重新编码音频
            return self._extract_audio_encode(output_file, config)
    
    def _extract_audio_copy(self, output_file: str, config: AudioExtractConfig) -> bool:
        """无损复制音频流"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vn',  # 禁用视频
            '-acodec', 'copy',  # 复制音频编码
            str(output_file),
            '-y',  # 覆盖输出文件
            '-v', 'quiet'
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def _extract_audio_encode(self, output_file: str, config: AudioExtractConfig) -> bool:
        """重新编码音频"""
        cmd = [
            'ffmpeg',
            '-i', str(self.video_path),
            '-vn',  # 禁用视频
        ]
        
        # 设置音频编码参数
        if config.audio_format == AudioFormat.MP3:
            cmd.extend(['-acodec', 'libmp3lame'])
            if config.quality == 'high':
                cmd.extend(['-q:a', '0'])  # 最高质量
            elif config.quality == 'medium':
                cmd.extend(['-q:a', '2'])
            else:
                cmd.extend(['-q:a', '5'])
        elif config.audio_format == AudioFormat.AAC:
            cmd.extend(['-acodec', 'aac'])
            if config.quality == 'high':
                cmd.extend(['-b:a', '320k'])
            elif config.quality == 'medium':
                cmd.extend(['-b:a', '192k'])
            else:
                cmd.extend(['-b:a', '128k'])
        elif config.audio_format == AudioFormat.FLAC:
            cmd.extend(['-acodec', 'flac'])
        elif config.audio_format == AudioFormat.WAV:
            cmd.extend(['-acodec', 'pcm_s16le'])
        elif config.audio_format == AudioFormat.OGG:
            cmd.extend(['-acodec', 'libvorbis'])
            if config.quality == 'high':
                cmd.extend(['-q:a', '10'])
            elif config.quality == 'medium':
                cmd.extend(['-q:a', '6'])
            else:
                cmd.extend(['-q:a', '3'])
        
        # 设置采样率和声道
        if config.sample_rate:
            cmd.extend(['-ar', str(config.sample_rate)])
        if config.channels:
            cmd.extend(['-ac', str(config.channels)])
        
        cmd.extend([
            str(output_file),
            '-y',
            '-v', 'quiet'
        ])
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

class MultiAudioExtractor:
    """多音频批量提取器"""
    
    def __init__(self, max_workers: int = 4):
        """初始化批量提取器"""
        self.max_workers = max_workers
        self.results = {}
        self.errors = {}
    
    def _process_single_video(self, args: Tuple[str, str, AudioExtractConfig]) -> Tuple[str, str, Optional[str]]:
        """处理单个视频的音频提取"""
        video_path, output_dir, config = args
        try:
            extractor = FFmpegAudioExtractor(video_path)
            
            if not extractor.video_info['has_audio']:
                return video_path, "", "视频文件没有音频流"
            
            # 生成输出文件路径
            video_name = Path(video_path).stem
            output_file = Path(output_dir) / f"{video_name}.{config.audio_format.value}"
            
            # 提取音频
            success = extractor.extract_audio(str(output_file), config)
            
            if success:
                return video_path, str(output_file), None
            else:
                return video_path, "", "音频提取失败"
                
        except Exception as e:
            return video_path, "", str(e)
    
    def batch_extract(self,
                     video_paths: List[str],
                     output_dir: str = "extracted_audio",
                     config: AudioExtractConfig = None) -> Dict[str, str]:
        """
        批量提取音频
        
        Args:
            video_paths: 视频文件路径列表
            output_dir: 输出目录
            config: 音频提取配置
            
        Returns:
            Dict[str, str]: 视频路径到音频文件路径的映射
        """
        if config is None:
            config = AudioExtractConfig()
        
        print(f"开始批量提取音频: {len(video_paths)} 个视频，使用 {self.max_workers} 个工作线程")
        print(f"输出格式: {config.audio_format.value}, 质量: {config.quality}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 准备任务参数
        tasks = [(video_path, output_dir, config) for video_path in video_paths]
        
        results = {}
        errors = {}
        
        # 执行批量处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self._process_single_video, task): task[0]
                for task in tasks
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_path, audio_file, error = future.result()
                    if error:
                        errors[video_path] = error
                        print(f"❌ 提取失败 {Path(video_path).name}: {error}")
                    else:
                        if audio_file:  # 有音频文件生成
                            results[video_path] = audio_file
                            print(f"✅ 提取完成 {Path(video_path).name} -> {Path(audio_file).name}")
                        else:
                            errors[video_path] = "无音频流"
                            print(f"⚠️  无音频 {Path(video_path).name}")
                    
                    completed += 1
                    print(f"进度: {completed}/{len(video_paths)}")
                    
                except Exception as e:
                    errors[video_path] = str(e)
                    print(f"❌ 处理异常 {Path(video_path).name}: {e}")
        
        self.results = results
        self.errors = errors
        
        # 打印统计信息
        successful = len(results)
        failed = len(errors)
        
        print(f"\n音频提取完成!")
        print(f"成功: {successful} 个视频")
        print(f"失败: {failed} 个视频")
        print(f"输出目录: {output_dir}")
        
        if errors:
            print("\n错误详情:")
            for video_path, error in errors.items():
                print(f"  {Path(video_path).name}: {error}")
        
        return results


def find_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """在目录中查找视频文件"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    video_files = []
    path = Path(directory)
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            video_files.append(str(path))
    elif path.is_dir():
        for ext in extensions:
            video_files.extend([str(f) for f in path.rglob(f"*{ext}")])
    
    return sorted(list(set(video_files)))  # 去重并排序

def create_sample_configs():
    """创建示例配置"""
    # 视频分割配置
    segment_config = VideoSegmentConfig(
        window_duration=10.0,
        step_duration=5.0,
        copy_codec=True,
        max_workers=4,
        mode=SegmentMode.TIME
    )
    
    # 音频提取配置
    audio_config = AudioExtractConfig(
        audio_format=AudioFormat.MP3,
        quality="high",
        copy_codec=True,
        max_workers=4
    )
    
    return segment_config, audio_config

def main():
    """主函数示例"""
    try:
        # 方法1: 仅提取音频
        print("=== 批量音频提取 ===")
        video_directory = "/n/work1/muyun/SemiAutonomous/segment_video_w5_s3"  # 替换为您的视频目录
        video_paths = find_video_files(video_directory)
        
        if not video_paths:
            print("未找到视频文件")
            return
        
        print(f"找到 {len(video_paths)} 个视频文件")
        
        # 音频提取配置
        audio_config = AudioExtractConfig(
            audio_format=AudioFormat.AAC,
            quality="high",
            copy_codec=True,  # 无损复制，最快
            max_workers=6
        )
        
        # 批量提取音频
        audio_extractor = MultiAudioExtractor(max_workers=6)
        audio_results = audio_extractor.batch_extract(
            video_paths=video_paths,  # 处理前10个文件作为示例
            output_dir="/n/work1/muyun/SemiAutonomous/segment_audio_w5_s3",
            config=audio_config
        )
                
    except Exception as e:
        print(f"错误: {e}")

# 进度监控版本
class AdvancedAudioExtractor(MultiAudioExtractor):
    """高级音频提取器，支持进度监控"""
    
    def __init__(self, max_workers: int = 4):
        super().__init__(max_workers)
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def batch_extract_with_progress(self,
                                  video_paths: List[str],
                                  output_dir: str = "extracted_audio",
                                  config: AudioExtractConfig = None) -> Dict[str, str]:
        """带进度回调的批量音频提取"""
        if config is None:
            config = AudioExtractConfig()
        
        print(f"开始批量提取音频: {len(video_paths)} 个视频")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        tasks = [(video_path, output_dir, config) for video_path in video_paths]
        results = {}
        errors = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_video = {
                executor.submit(self._process_single_video, task): task[0]
                for task in tasks
            }
            
            completed = 0
            total = len(future_to_video)
            
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_path, audio_file, error = future.result()
                    if error:
                        errors[video_path] = error
                    else:
                        if audio_file:
                            results[video_path] = audio_file
                    
                    completed += 1
                    progress = completed / total
                    
                    # 调用进度回调
                    if self.progress_callback:
                        self.progress_callback(completed, total, progress, video_path, audio_file)
                    else:
                        status = "✅" if audio_file and not error else "❌"
                        print(f"{status} [{completed}/{total}] {progress:.1%} - {Path(video_path).name}")
                        
                except Exception as e:
                    errors[video_path] = str(e)
        
        self.results = results
        self.errors = errors
        return results

def audio_progress_callback(completed: int, total: int, progress: float, 
                          video_path: str, audio_file: str):
    """音频提取进度回调示例"""
    status = "✅" if audio_file else "❌"
    audio_name = Path(audio_file).name if audio_file else "失败"
    print(f"{status} [{completed}/{total}] {progress:.1%} - "
          f"{Path(video_path).name} -> {audio_name}")

if __name__ == "__main__":
    main()