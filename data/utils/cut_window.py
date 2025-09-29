import subprocess
import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time
from dataclasses import dataclass
from enum import Enum

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

class FFmpegVideoSegmenter:
    def __init__(self, video_path: str):
        """初始化FFmpeg视频分割器"""
        self.video_path = Path(video_path)
        self.lock = threading.Lock()  # 线程安全锁
        
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
            
            # 提取视频流信息
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("未找到视频流")
            
            # 计算FPS
            fps_str = video_stream.get('avg_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str) if fps_str else 0
            
            return {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps,
                'duration': float(info['format'].get('duration', 0)),
                'codec': video_stream.get('codec_name', ''),
                'bit_rate': info['format'].get('bit_rate', '0')
            }
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"获取视频信息失败: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"解析视频信息失败: {e}")
    
    def seconds_to_timestamp(self, seconds: float) -> str:
        """将秒数转换为时间戳格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _extract_segment(self, start_time: float, end_time: float, 
                        output_file: str, copy_codec: bool = True) -> bool:
        """提取视频段"""
        duration = end_time - start_time
        
        if copy_codec:
            # 使用流复制，无损且快速
            cmd = [
                'ffmpeg',
                '-i', str(self.video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-reset_timestamps', '1',
                str(output_file),
                '-y',
                '-v', 'quiet'  # 减少输出信息
            ]
        else:
            # 重新编码
            cmd = [
                'ffmpeg',
                '-i', str(self.video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'medium',
                '-crf', '18',
                str(output_file),
                '-y',
                '-v', 'quiet'
            ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False
    
    def segment_by_time(self,
                       window_duration: float,
                       step_duration: float,
                       output_dir: str = "segments",
                       copy_codec: bool = True,
                       prefix: str = "segment") -> List[str]:
        """按时间窗口分割视频"""
        if window_duration <= 0 or step_duration <= 0:
            raise ValueError("窗口大小和步长必须大于0")
        
        if window_duration > self.video_info['duration']:
            raise ValueError("窗口时长大于视频总时长")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        video_ext = self.video_path.suffix
        video_name = "_".join(self.video_path.stem.split("_")[:-3])
        
        generated_files = []
        segment_index = 0
        start_time = 0.0
        
        while start_time < self.video_info['duration']:
            end_time = min(start_time + window_duration, self.video_info['duration'])
            
            # 生成输出文件名
            start_timestamp = self.seconds_to_timestamp(start_time)
            end_timestamp = self.seconds_to_timestamp(end_time)
            
            clean_start = re.sub(r'[^\d.]', '', start_timestamp.replace(':', ''))
            clean_end = re.sub(r'[^\d.]', '', end_timestamp.replace(':', ''))
            
            output_filename = f"{video_name}_{segment_index:02d}_{clean_start}_to_{clean_end}{video_ext}"
            output_file = output_path / output_filename
            
            # 分割视频段
            success = self._extract_segment(start_time, end_time, str(output_file), copy_codec)
            
            if success:
                generated_files.append(str(output_file))
            
            segment_index += 1
            start_time += step_duration
            
            if start_time >= self.video_info['duration']:
                break
        
        return generated_files

class MultiVideoSegmenter:
    """多视频批量分割器"""
    
    def __init__(self, max_workers: int = 4, use_process: bool = False):
        """
        初始化批量分割器
        
        Args:
            max_workers: 最大工作线程数
            use_process: 是否使用进程池（CPU密集型任务）
        """
        self.max_workers = max_workers
        self.use_process = use_process
        self.results = {}
        self.errors = {}
        
    def _process_single_video(self, args: Tuple[str, VideoSegmentConfig]) -> Tuple[str, List[str], Optional[str]]:
        """处理单个视频"""
        video_path, config = args
        try:
            segmenter = FFmpegVideoSegmenter(video_path)
            
            # 根据模式选择分割方法
            if config.mode == SegmentMode.TIME:
                segments = segmenter.segment_by_time(
                    window_duration=config.window_duration,
                    step_duration=config.step_duration,
                    output_dir=Path(config.output_dir).joinpath(f"segments_{Path(video_path).stem}"),
                    copy_codec=config.copy_codec,
                    prefix=config.prefix
                )
            elif config.mode == SegmentMode.OVERLAP:
                # 实现重叠分割
                step_duration = config.window_duration - config.overlap_duration
                segments = segmenter.segment_by_time(
                    window_duration=config.window_duration,
                    step_duration=step_duration,
                    output_dir=Path(config.output_dir).joinpath(f"segments_overlap_{Path(video_path).stem}"),
                    copy_codec=config.copy_codec,
                    prefix="overlap"
                )
            else:
                # 帧数分割
                if segmenter.video_info['fps'] <= 0:
                    raise ValueError("无法获取视频FPS信息")
                window_duration = config.window_frames / segmenter.video_info['fps']
                step_duration = config.step_frames / segmenter.video_info['fps']
                segments = segmenter.segment_by_time(
                    window_duration=window_duration,
                    step_duration=step_duration,
                    output_dir=Path(config.output_dir).joinpath(f"segments_frames_{Path(video_path).stem}"),
                    copy_codec=config.copy_codec,
                    prefix="frame"
                )
            
            return video_path, segments, None
            
        except Exception as e:
            return video_path, [], str(e)
    
    def batch_segment(self, 
                     video_paths: List[str], 
                     config: VideoSegmentConfig) -> Dict[str, List[str]]:
        """
        批量分割多个视频
        
        Args:
            video_paths: 视频文件路径列表
            config: 分割配置
            
        Returns:
            Dict[str, List[str]]: 每个视频的分割结果
        """
        print(f"开始批量处理 {len(video_paths)} 个视频，使用 {self.max_workers} 个工作线程")
        
        # 准备任务参数
        tasks = [(video_path, config) for video_path in video_paths]
        
        # 选择执行器类型
        if self.use_process:
            executor_class = ProcessPoolExecutor
            print("使用进程池执行器")
        else:
            executor_class = ThreadPoolExecutor
            print("使用线程池执行器")
        
        results = {}
        errors = {}
        
        # 执行批量处理
        with executor_class(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_video = {
                executor.submit(self._process_single_video, task): task[0] 
                for task in tasks
            }
            
            # 收集结果
            completed = 0
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_path, segments, error = future.result()
                    if error:
                        errors[video_path] = error
                        print(f"❌ 处理失败 {video_path}: {error}")
                    else:
                        results[video_path] = segments
                        print(f"✅ 处理完成 {video_path}: 生成 {len(segments)} 个片段")
                    
                    completed += 1
                    print(f"进度: {completed}/{len(video_paths)}")
                    
                except Exception as e:
                    errors[video_path] = str(e)
                    print(f"❌ 处理异常 {video_path}: {e}")
        
        self.results = results
        self.errors = errors
        
        # 打印统计信息
        successful = len(results)
        failed = len(errors)
        total_segments = sum(len(segments) for segments in results.values())
        
        print(f"\n批量处理完成!")
        print(f"成功: {successful} 个视频")
        print(f"失败: {failed} 个视频")
        print(f"总计生成片段: {total_segments} 个")
        
        if errors:
            print("\n错误详情:")
            for video_path, error in errors.items():
                print(f"  {video_path}: {error}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """获取处理统计信息"""
        total_videos = len(self.results) + len(self.errors)
        successful_videos = len(self.results)
        failed_videos = len(self.errors)
        total_segments = sum(len(segments) for segments in self.results.values())
        
        return {
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'failed_videos': failed_videos,
            'total_segments': total_segments,
            'success_rate': successful_videos / total_videos if total_videos > 0 else 0
        }

def create_sample_config() -> VideoSegmentConfig:
    """创建示例配置"""
    return VideoSegmentConfig(
        window_duration=10.0,
        step_duration=5.0,
        window_frames=300,
        step_frames=150,
        overlap_duration=2.0,
        copy_codec=True,
        prefix="segment",
        max_workers=4,
        mode=SegmentMode.TIME
    )

def find_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """在目录中查找视频文件"""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    video_files = []
    path = Path(directory)
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            video_files.append(str(path))
    elif path.is_dir():
        for ext in extensions:
            video_files.extend([str(f) for f in path.rglob(f"*{ext}")])
    
    return sorted(video_files)


def find_audio_files(directory: str, extensions: List[str] = None) -> List[str]:
    """在目录中查找视频文件"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    
    video_files = []
    path = Path(directory)
    
    if path.is_file():
        if path.suffix.lower() in extensions:
            video_files.append(str(path))
    elif path.is_dir():
        for ext in extensions:
            video_files.extend([str(f) for f in path.rglob(f"*{ext}")])
    
    return sorted(video_files)


def main():
    """主函数示例"""
    try:
        # 方法1: 指定视频文件列表
        video_dir = "/n/work1/muyun/SemiAutonomous/videofiles_pure"

        video_paths = [x for x in Path(video_dir).glob("*.mp4")]
        
        # 过滤存在的文件
        video_paths = [path for path in video_paths if Path(path).exists()]
        
        if not video_paths:
            print("没有找到有效的视频文件")
            return
        
        # 创建配置
        config = VideoSegmentConfig(
            window_duration=5,    # 10秒窗口
            step_duration=3,      # 5秒步长
            copy_codec=True,         # 无损流复制
            max_workers=4,           # 4个工作线程
            mode=SegmentMode.OVERLAP,
            output_dir="/n/work1/muyun/SemiAutonomous/segment_video_w5_s3"
        )
        
        # 创建批量分割器
        batch_segmenter = MultiVideoSegmenter(max_workers=config.max_workers)
        
        # 执行批量分割
        results = batch_segmenter.batch_segment(video_paths, config)
        
        # 获取统计信息
        stats = batch_segmenter.get_statistics()
        print(f"\n统计信息: {stats}")
        
        # # 方法2: 从目录查找视频文件
        # print("\n=== 从目录查找视频文件 ===")
        # video_directory = "videos/"  # 替换为您的视频目录
        # found_videos = find_video_files(video_directory)
        
        # if found_videos:
        #     print(f"找到 {len(found_videos)} 个视频文件")
        #     for video in found_videos[:5]:  # 只显示前5个
        #         print(f"  - {video}")
        #     if len(found_videos) > 5:
        #         print(f"  ... 还有 {len(found_videos) - 5} 个文件")
            
        #     # 批量处理找到的视频
        #     config2 = VideoSegmentConfig(
        #         window_duration=10.0,
        #         step_duration=5.0,
        #         copy_codec=True,
        #         max_workers=6,  # 增加工作线程数
        #         mode=SegmentMode.TIME
        #     )
            
        #     batch_segmenter2 = MultiVideoSegmenter(max_workers=config2.max_workers)
        #     results2 = batch_segmenter2.batch_segment(found_videos[:10], config2)  # 处理前10个
            
        # else:
        #     print("在指定目录中没有找到视频文件")
            
    except Exception as e:
        print(f"错误: {e}")


# 高级批量处理类
class AdvancedBatchSegmenter(MultiVideoSegmenter):
    """高级批量分割器，支持更多功能"""
    
    def __init__(self, max_workers: int = 4, use_process: bool = False):
        super().__init__(max_workers, use_process)
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def batch_segment_with_progress(self, 
                                  video_paths: List[str], 
                                  config: VideoSegmentConfig) -> Dict[str, List[str]]:
        """带进度回调的批量分割"""
        print(f"开始批量处理 {len(video_paths)} 个视频...")
        
        tasks = [(video_path, config) for video_path in video_paths]
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
                    video_path, segments, error = future.result()
                    if error:
                        errors[video_path] = error
                    else:
                        results[video_path] = segments
                    
                    completed += 1
                    progress = completed / total
                    
                    # 调用进度回调
                    if self.progress_callback:
                        self.progress_callback(completed, total, progress, video_path)
                    else:
                        print(f"进度: {completed}/{total} ({progress:.1%})")
                        
                except Exception as e:
                    errors[video_path] = str(e)
        
        self.results = results
        self.errors = errors
        return results

def progress_callback(completed: int, total: int, progress: float, current_video: str):
    """进度回调示例"""
    print(f"[{completed}/{total}] {progress:.1%} - 完成: {Path(current_video).name}")

if __name__ == "__main__":
    main()