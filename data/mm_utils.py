import ast
import os
import re
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import torch
import torchvision.transforms.functional as VF
import numpy as np
from transformers import StoppingCriteria

import cv2
import imageio

import os
os.environ["PATH"] = "/misc/home/muyun/VScode/project/LLM/SilenceForStreaming/ffmpeg-master-latest-linux64-gpl/bin" + os.pathsep + os.environ["PATH"]

import ffmpeg
from PIL import Image
from decord import VideoReader, cpu

import random
import librosa
from scipy.io import wavfile
import soundfile as sf
import torchaudio.compliance.kaldi as ta_kaldi


def process_audio_file(wav_path):
    # read wav
    #print(wav_path)
    wav, sr = sf.read(wav_path)
    if len(wav.shape) == 2:
        wav = wav[:, 0]
    if len(wav) > 30 * sr:
        max_start = len(wav) - 30 * sr
        start = random.randint(0, max_start)
        wav = wav[start: start + 30 * sr]
    if len(wav) < 30 * sr:
        pad_length = 30 * sr - len(wav)
        wav = np.pad(wav, (0, pad_length), mode='constant', constant_values=0.0)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")

    # beats
    raw_wav = torch.from_numpy(wav).to('cpu')
    waveform = raw_wav.unsqueeze(0) * 2 ** 15
    fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10).to(torch.float32)
    return fbank.unsqueeze(0)


def process_audio(wav_path, processor):
    # wav, sr = sf.read(wav_path)
    # if len(wav.shape) == 2:
    #     wav = wav[:, 0]
    # if len(wav) > 30 * sr:
    #     max_start = len(wav) - 30 * sr
    #     start = random.randint(0, max_start)
    #     wav = wav[start: start + 30 * sr]
    # if len(wav) < 30 * sr:
    #     pad_length = 30 * sr - len(wav)
    #     wav = np.pad(wav, (0, pad_length), mode='constant', constant_values=0.0)
    # if sr != 16000:
    #     wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")
    sr, wav = wavfile.read(wav_path)
    assert sr == 16_000 and len(wav.shape) == 1
    # raw_wav = torch.from_numpy(wav).to('cpu')
    audio_inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
    audio_feats = audio_inputs.input_features if hasattr(audio_inputs, "input_features") else audio_inputs.input_values
    return audio_feats.to(torch.float)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        # NOTE: v1 version
        # Calculate the size of each segment from which a frame will be extracted
        # if duration <= num_frames:
        #     return np.arange(duration).astype(int)
        # seg_size = float(duration - 1) / num_frames

        # frame_ids = []
        # for i in range(num_frames):
        #     # Calculate the start and end indices of each segment
        #     start = seg_size * i
        #     end   = seg_size * (i + 1)
        #     # Append the middle index of the segment to the list
        #     frame_ids.append((start + end) / 2)

        # return np.round(np.array(frame_ids) + 1e-6).astype(int)
        # NOTE: v0 version
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=None, temporal_factor=1):
    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    # 1. Loading Video
    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))

        vid_fps = 3
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)

        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        # vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    # 2. Determine frame range & Calculate frame indices
    f_start = 0                       if s is None else max(int(s * vid_fps) - 1, 0)
    f_end   = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    # 3. Sampling frame indices
    sampled_frame_indices = frame_indices

    # 4. Acquire frame data
    if os.path.isdir(video_path):
        frames = [cv2.cvtColor(cv2.imread(os.path.join(video_path, frame_files[frame_idx])), cv2.COLOR_BGR2RGB) for frame_idx in sampled_frame_indices]
    elif video_path.endswith('.gif'):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    # frames = frames.transpose(0, 3, 1, 2)
    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    # NOTE: pad the video with black frames
    # while num_frames is not None and len(video_data) < num_frames:
    #     video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))

    return frames, timestamps


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1
):
    """
    Load and process a video file and return the frames and the timestamps of each frame.

    Args:
        video_path (str): Path to the video file.
        start_time (float, optional): Start time in seconds. Defaults to None.
        end_time (float, optional): End time in seconds. Defaults to None.
        fps (float, optional): Frames per second. Defaults to None.
        num_frames (float, optional): Number of frames to sample. Defaults to None.
        size (int, optional): Size of the shortest side. Defaults to None.
        size_divisible (int, optional): Size divisible by this number. Defaults to 1.
        precise_time (bool, optional): Whether to use precise time. Defaults to False.
        verbose (bool, optional): Print ffmpeg output. Defaults to False.

    Returns:
        frames (List[PIL.Image]): List of frames.
        timestamps (List[float]): List of timestamps.
    """
    if start_time is not None and end_time is not None and end_time - start_time < 1:
        return load_video_from_ids(video_path, start_time, end_time, fps=fps)
    if os.path.isdir(video_path):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps)
    if video_path.endswith('.gif'):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps)
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w, h = int(video_stream['width']), int(video_stream['height'])

    kwargs, input_kwargs, output_kwargs = {}, {}, {}
    do_trim = start_time is not None or end_time is not None
    if start_time is not None:
        new_start_time = max(float(video_stream['start_time']), start_time)
        duration -= new_start_time - start_time
        start_time = new_start_time
    else:
        start_time = float(video_stream['start_time'])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    else:
        duration = duration
    if do_trim:
        kwargs = {'ss': start_time, 't': duration}
    if precise_time:
        output_kwargs.update(kwargs)
    else:
        input_kwargs.update(kwargs)

    if size is not None:
        scale_factor = size / min(w, h)
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
    else:
        new_w, new_h = w, h
    new_w = new_w // size_divisible * size_divisible
    new_h = new_h // size_divisible * size_divisible

    # NOTE: It may result in unexpected number of frames in ffmpeg
    # if calculate the fps directly according to max_frames
    # NOTE: the below lines may hurt the performance
    # if max_frames is not None and (fps is None or duration * fps > 2 * max_frames):
    #     fps = max_frames / duration * 2

    stream = ffmpeg.input(video_path, **input_kwargs)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if new_w != w or new_h != h:
        stream = ffmpeg.filter(stream, 'scale', new_w, new_h)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", **output_kwargs)
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)

    frames = np.frombuffer(out, np.uint8).reshape([-1, new_h, new_w, 3]).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(start_time, start_time + duration + 1 / fps, 1 / fps)[:len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))


    return frames, timestamps    


def process_video(video_path, processor, s=None, e=None, aspect_ratio='pad', num_frames=None):
    if isinstance(video_path, str):
        if s is not None and e is not None:
            s = s if s >= 0. else 0.
            e = e if e >= 0. else 0.
            if s > e:
                s, e = e, s
            elif s == e:
                e = s + 1
        
        # 1. Loading Video
        if os.path.isdir(video_path):                
            frame_files = sorted(os.listdir(video_path))

            fps = 3
            num_frames_of_video = len(frame_files)
        elif video_path.endswith('.gif'):
            gif_reader = imageio.get_reader(video_path)

            fps = 25
            num_frames_of_video = len(gif_reader)
        else:
            vreader = VideoReader(video_path, ctx=cpu(0), num_threads=1)

            fps = vreader.get_avg_fps()
            num_frames_of_video = len(vreader)

        # 2. Determine frame range & Calculate frame indices
        f_start = 0                       if s is None else max(int(s * fps) - 1, 0)
        f_end   = num_frames_of_video - 1 if e is None else min(int(e * fps) - 1, num_frames_of_video - 1)
        frame_indices = list(range(f_start, f_end + 1))

        duration = len(frame_indices)
        # 3. Sampling frame indices 
        if num_frames is None:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=fps, fps=fps)]
        else:
            sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=num_frames)]

        # 4. Acquire frame data
        if os.path.isdir(video_path): 
            video_data = [Image.open(os.path.join(video_path, frame_files[f_idx])) for f_idx in sampled_frame_indices]
        elif video_path.endswith('.gif'):
            video_data = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
        else:
            video_data = [Image.fromarray(frame) for frame in vreader.get_batch(sampled_frame_indices).asnumpy()]

    elif isinstance(video_path, np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], np.ndarray):
        video_data = [Image.fromarray(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], str):
        video_data = [Image.open(f) for f in video_path]
    elif isinstance(video_path, list) and isinstance(video_path[0], Image.Image):
        video_data = video_path
    else:
        raise ValueError(f"Unsupported video path type: {type(video_path)}")

    while num_frames is not None and len(video_data) < num_frames:
        video_data.append(Image.fromarray(np.zeros((*video_data[-1].size, 3), dtype=np.uint8)))


    if aspect_ratio == 'pad':
        images = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values'].unsqueeze(0)
    else:
        images = [f for f in video_data]
        video = processor.preprocess(images, return_tensors='pt')['pixel_values'].unsqueeze(0)
        
    return video


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)