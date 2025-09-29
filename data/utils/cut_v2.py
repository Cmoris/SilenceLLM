import json
import numpy as np
from pathlib import Path
import soundfile as sf
import librosa
import argparse
import subprocess
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def process_segment(args):
    """
    处理单个长沉默段，生成音频和视频片段
    """
    (
        idx,
        silence,
        audio,
        sr,
        total_duration,
        audio_file,
        name,
        video_file,
        audio_output_dir,
        video_output_dir,
        min_silence_duration
    ) = args

    silence_start = silence['start']
    silence_end = silence['end']
    silence_duration = silence_end - silence_start
    label = [silence['Stopped'], silence["Thinking"]]

    start = random.uniform(2, 8)
    end = 10 - start
    segment_start = max(0.0, silence_start - start)
    segment_end = min(total_duration, silence_start + end)
    segment_duration = segment_end - segment_start

    start_idx = int(segment_start * sr)
    end_idx = int(segment_end * sr)
    segment_audio = audio[start_idx:end_idx]

    audio_output_filename = f"{name}_{idx}{Path(audio_file).suffix}"
    audio_output_path = audio_output_dir / audio_output_filename
    sf.write(str(audio_output_path), segment_audio, sr)

    video_output_path = None
    if video_file:
        video_output_filename = f"{name}_{idx}{video_file.suffix}"
        video_output_path = video_output_dir / video_output_filename

        cmd = [
            'ffmpeg', '-i', str(video_file),
            '-ss', str(segment_start),
            '-t', str(segment_duration),
            '-c:v', 'libx264', '-c:a', 'aac',
            str(video_output_path), '-y'
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            cmd = [
                'ffmpeg', '-i', str(video_file),
                '-ss', str(segment_start),
                '-t', str(segment_duration),
                '-c:v', 'libx264', '-c:a', 'aac',
                str(video_output_path), '-y'
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"视频处理失败: {e}")
                video_output_path = None

    return {
        "audio_path": str(audio_output_path),
        "video_path": str(video_output_path) if video_output_path else None,
        "start": silence_start - segment_start,
        "end": min(silence_end - segment_start, segment_end - segment_start),
        "label": label
    }

def split_long_silence_segments(jsonl_file, audio_output_dir, video_dir, video_output_dir, min_silence_duration=2.0, max_workers=4):
    if not check_ffmpeg():
        print("错误: 未找到ffmpeg，请先安装ffmpeg")
        sys.exit(1)

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
    results = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                filename = data.get('filename', '')
                filepath = data.get('filepath', '')
                silence_segments = data.get('silence_segments', [])

                if not filename or not filepath or not silence_segments:
                    continue

                print(f"\n处理文件 {line_num}: {filename}")

                if not Path(filepath).exists():
                    print(f"  跳过: 原音频文件不存在")
                    continue

                audio_stem = filename
                video_stem = filename + "_audio_video_subject"
                name = filename
                video_extensions = ['.mp4']
                video_file = None

                for ext in video_extensions:
                    potential_video = video_dir / (video_stem + ext)
                    if potential_video.exists():
                        video_file = potential_video
                        break
                           
                if not video_file:
                    print(f"  警告: 未找到对应的视频文件 ({video_stem}.[视频格式])")

                audio, sr = librosa.load(filepath, sr=None)
                total_duration = len(audio) / sr
                print(f"  音频时长: {total_duration:.2f} 秒")

                long_silences = [seg for seg in silence_segments if seg['end'] - seg['start'] > min_silence_duration]
                labels = []

                if not long_silences:
                    print(f"  没有找到大于 {min_silence_duration} 秒的沉默段")
                    continue

                print(f"  发现 {len(long_silences)} 个长沉默段")

                tasks = [
                    (
                        idx + 1,
                        silence,
                        audio,
                        sr,
                        total_duration,
                        filepath,
                        name,
                        video_file,
                        audio_output_dir,
                        video_output_dir,
                        min_silence_duration
                    )
                    for idx, silence in enumerate(long_silences)
                ]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_segment, task) for task in tasks]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            results.append(result)

                file_count += 1

            except json.JSONDecodeError:
                print(f"  错误: 第 {line_num} 行不是有效的JSON")
                continue
            except Exception as e:
                print(f"  错误: 处理第 {line_num} 行时出错: {e}")
                continue

    # 写入新的 JSONL 文件
    output_jsonl = audio_output_dir / "output_segments.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"\n" + "="*50)
    print(f"处理完成!")
    print(f"处理文件数: {file_count}")
    print(f"创建片段数: {len(results)}")
    print(f"输出JSONL文件: {output_jsonl}")
    print(f"音频输出目录: {audio_output_dir.absolute()}")
    print(f"视频输出目录: {video_output_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser(description='根据长沉默段分割音频和对应的视频文件')
    parser.add_argument('--jsonl_file', required=True, help='沉默检测结果文件')
    parser.add_argument('--audio_output_dir', required=True, help='音频分割输出目录')
    parser.add_argument('--video_dir', required=True, help='原始视频文件目录')
    parser.add_argument('--video_output_dir', required=True, help='视频分割输出目录')
    parser.add_argument('--min_silence', type=float, default=2.0, help='最小沉默时长(秒)')
    parser.add_argument('--max_workers', type=int, default=4, help='最大线程数 (默认: 4)')

    args = parser.parse_args()

    split_long_silence_segments(
        args.jsonl_file,
        args.audio_output_dir,
        args.video_dir,
        args.video_output_dir,
        min_silence_duration=args.min_silence,
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    main()