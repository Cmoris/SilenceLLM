import json
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import argparse
import subprocess
import sys
import random

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def split_long_silence_segments(jsonl_file, audio_output_dir, video_dir, video_output_dir, min_silence_duration=2.0):
    """
    分割长沉默段并同时裁剪对应的视频
    
    Args:
        jsonl_file: 包含沉默检测结果的JSONL文件
        audio_output_dir: 音频分割输出目录
        video_dir: 原始视频文件目录
        video_output_dir: 视频分割输出目录
        min_silence_duration: 最小沉默持续时间（秒）
    """
    # 检查ffmpeg
    if not check_ffmpeg():
        print("错误: 未找到ffmpeg，请先安装ffmpeg")
        sys.exit(1)
    
    # 创建输出目录
    audio_output_dir = Path(audio_output_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = Path(video_dir)
    video_output_dir = Path(video_output_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始处理: {jsonl_file}")
    print(f"音频输出目录: {audio_output_dir}")
    print(f"视频目录: {video_dir}")
    print(f"视频输出目录: {video_output_dir}")
    print(f"最小沉默时长: {min_silence_duration} 秒")
    
    file_count = 0
    audio_segment_count = 0
    video_segment_count = 0
    
    # 读取JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析数据
                data = json.loads(line.strip())
                filename = data.get('filename', '')
                filepath = data.get('filepath', '')
                silence_segments = data.get('silence_segments', [])
                
                if not filename or not filepath or not silence_segments:
                    continue
                
                print(f"\n处理文件 {line_num}: {filename}")
                
                # 检查原音频文件是否存在
                if not Path(filepath).exists():
                    print(f"  跳过: 原音频文件不存在")
                    continue
                
                # 查找对应的视频文件
                audio_stem = Path(filename).stem
                video_stem = "_".join(audio_stem.split('_')[:-1]) + "_video_subject"
                name = '_'.join(audio_stem.split('_')[:-2])
                video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
                video_file = None
                
                for ext in video_extensions:
                    potential_video = video_dir / (video_stem + ext)
                    if potential_video.exists():
                        video_file = potential_video
                        break
                
                if not video_file:
                    print(f"  警告: 未找到对应的视频文件 ({video_stem}.[视频格式])")
                
                # 加载音频
                audio, sr = librosa.load(filepath, sr=None)
                total_duration = len(audio) / sr
                print(f"  音频时长: {total_duration:.2f} 秒")
                
                # 筛选长沉默段
                long_silences = [
                    seg for seg in silence_segments 
                    if seg['end'] - seg['start'] > min_silence_duration
                ]
                
                if not long_silences:
                    print(f"  没有找到大于 {min_silence_duration} 秒的沉默段")
                    continue
                
                print(f"  发现 {len(long_silences)} 个长沉默段")
                
                # 处理每个长沉默段
                for idx, silence in enumerate(long_silences, 1):
                    silence_start = silence['start']
                    silence_end = silence['end']
                    silence_duration = silence_end - silence_start
                    
                    # 计算分割边界（沉默前5秒到沉默后5秒）
                    start = random.uniform(2, 8)
                    end = 10 - start
                    segment_start = max(0.0, silence_start - start)
                    segment_end = min(total_duration, silence_start + end)
                    segment_duration = segment_end - segment_start
                    
                    # 分割音频
                    start_idx = int(segment_start * sr)
                    end_idx = int(segment_end * sr)
                    segment_audio = audio[start_idx:end_idx]
                    
                    # 生成音频输出文件名
                    # audio_output_filename = f"{audio_stem}_silence_{idx}_{silence_start:.1f}-{silence_end:.1f}s{Path(filename).suffix}"
                    audio_output_filename = f"{name}_{idx}{Path(filename).suffix}"
                    audio_output_path = audio_output_dir / audio_output_filename
                    
                    # 保存音频片段
                    sf.write(str(audio_output_path), segment_audio, sr)
                    print(f"    已保存音频: {audio_output_filename} ({segment_duration:.2f} 秒)--{start}-{end}")
                    audio_segment_count += 1
                    
                    # 如果有对应的视频文件，则分割视频
                    if video_file:
                        # video_output_filename = f"{video_stem}_silence_{idx}_{silence_start:.1f}-{silence_end:.1f}s{video_file.suffix}"
                        video_output_filename = f"{name}_{idx}{video_file.suffix}"
                        video_output_path = video_output_dir / video_output_filename
                        
                        # 使用ffmpeg裁剪视频
                        cmd = [
                                'ffmpeg',
                                '-i', str(video_file),
                                '-ss', str(segment_start),
                                '-t', str(segment_duration),
                                '-c:v', 'libx264',
                                '-c:a', 'aac',
                                str(video_output_path),
                                '-y'
                            ]
                        
                        try:
                            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                            print(f"    已保存视频: {video_output_filename}--{start}-{end}")
                            video_segment_count += 1
                        except subprocess.CalledProcessError as e:
                            print(f"    视频分割失败: {e}")
                            # 如果copy失败，尝试重新编码
                            cmd = [
                                'ffmpeg',
                                '-i', str(video_file),
                                '-ss', str(segment_start),
                                '-t', str(segment_duration),
                                '-c:v', 'libx264',
                                '-c:a', 'aac',
                                str(video_output_path),
                                '-y'
                            ]
                            try:
                                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                                print(f"    已保存视频(重新编码): {video_output_filename}")
                                video_segment_count += 1
                            except subprocess.CalledProcessError as e2:
                                print(f"    视频分割仍失败: {e2}")
                
                file_count += 1
                
            except json.JSONDecodeError:
                print(f"  错误: 第 {line_num} 行不是有效的JSON")
                continue
            except Exception as e:
                print(f"  错误: 处理第 {line_num} 行时出错: {e}")
                continue
    
    print(f"\n" + "="*50)
    print(f"处理完成!")
    print(f"处理文件数: {file_count}")
    print(f"创建音频片段数: {audio_segment_count}")
    print(f"创建视频片段数: {video_segment_count}")
    print(f"音频输出目录: {audio_output_dir.absolute()}")
    print(f"视频输出目录: {video_output_dir.absolute()}")

def create_processing_report(jsonl_file, audio_output_dir, video_dir, video_output_dir):
    """
    创建处理报告
    """
    report = {
        "source_file": str(jsonl_file),
        "audio_output_directory": str(audio_output_dir),
        "video_directory": str(video_dir),
        "video_output_directory": str(video_output_dir),
        "processed_files": 0,
        "created_audio_segments": 0,
        "created_video_segments": 0,
        "file_details": []
    }
    
    # 统计输出目录中的文件
    audio_files = list(Path(audio_output_dir).glob("*_silence_*"))
    video_files = list(Path(video_output_dir).glob("*_silence_*"))
    
    report["created_audio_segments"] = len(audio_files)
    report["created_video_segments"] = len(video_files)
    
    # 读取JSONL文件统计
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                report["file_details"].append({
                    "filename": data.get("filename", ""),
                    "silence_segments_count": len(data.get("silence_segments", []))
                })
                report["processed_files"] += 1
            except:
                continue
    
    # 保存报告
    report_file = Path(audio_output_dir) / "processing_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"处理报告已保存到: {report_file}")
    return report

def main():
    parser = argparse.ArgumentParser(description='根据长沉默段分割音频和对应的视频文件')
    parser.add_argument('--jsonl_file', required=True, help='沉默检测结果文件')
    parser.add_argument('--audio_output_dir', required=True, help='音频分割输出目录')
    parser.add_argument('--video_dir', required=True, help='原始视频文件目录')
    parser.add_argument('--video_output_dir', required=True, help='视频分割输出目录')
    parser.add_argument('--min_silence', type=float, default=2.0, help='最小沉默时长(秒) (默认: 2.0)')
    
    args = parser.parse_args()
    
    # 分割音频和视频
    split_long_silence_segments(
        args.jsonl_file,
        args.audio_output_dir,
        args.video_dir,
        args.video_output_dir,
        min_silence_duration=args.min_silence
    )
    
    # 创建报告
    create_processing_report(
        args.jsonl_file,
        args.audio_output_dir,
        args.video_dir,
        args.video_output_dir
    )

if __name__ == "__main__":
    main()