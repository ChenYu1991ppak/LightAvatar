import os
import cv2
import torch
import numpy as np
from functools import partial
from PIL import Image
from tqdm import tqdm

from tools.common import load_exp_embs
from model.hparams import hparams as hp


VIDEO_PATCH = lambda x: 'exp_{}.mp4'.format(x)
IMAGE_PATCH = lambda x: 'frame_{}.jpg'.format(x)


class SearchFramesByAudioSegment(object):
    """ 从合成图像中查找与音频匹配的图像，合成图像由原视频逐帧推理得到
        合成图像总数 = 离散口型总数M * 原运动视频总帧数N；即合成图像由M个视频片段组成，每个视频片段包含相同口型N帧
    """

    def __init__(self,
                 src_dir,  # 视频分片目录
                 cluster_path,  # 离散口型索引文件路径
                 batch_size: int = 8
                 ):
        assert os.path.exists(cluster_path), f"Missing file: {cluster_path}"

        self.exp_embs_tensor = torch.tensor(load_exp_embs(cluster_path), dtype=torch.float32)
        self.exp_num = self.exp_embs_tensor.shape[0]
        self.fps = hp.fps
        self.batch_size = batch_size
        self.frame_loader = self._init_frame_loader(src_dir)

    def __call__(self, aud_embs, motion_idx):
        exp_idx_seq = self.find_exp(aud_embs)
        yield from self.frame_loader(exp_idx_seq=exp_idx_seq, motion_idx=motion_idx)

    def _init_frame_loader(self, src_dir):
        if src_dir.endswith('_video'):
            cap = cv2.VideoCapture(os.path.join(src_dir, VIDEO_PATCH(0)))
            round_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            frame_loader = partial(_load_frames_from_videos, src_dir=src_dir, round_size=round_size)
        else:
            round_size = len([f for f in os.listdir(src_dir)]) // self.exp_num
            frame_loader = partial(_load_frames_from_images, src_dir=src_dir, round_size=round_size)
        return frame_loader

    def find_exp(self, audio_embs):
        with torch.no_grad():
            audio_embs_tensor = torch.tensor(audio_embs, device='cpu', dtype=torch.float32)
            distances = torch.cdist(audio_embs_tensor, self.exp_embs_tensor, compute_mode='donot_use_mm_for_euclid_dist')
            exp_idx_seq = torch.topk(distances, 1, largest=False, sorted=False, dim=1).indices.squeeze()
        return list(exp_idx_seq)


def _load_frames_from_images(src_dir, exp_idx_seq, motion_idx, round_size):
    def read_frame(i):
        midx = (motion_idx + i) % (2 * round_size)
        midx = midx if midx < round_size else 2 * round_size - 1 - midx  # 往复循环
        frame_id = exp_idx_seq[i] * round_size + midx
        image_path = os.path.join(src_dir, IMAGE_PATCH(frame_id))
        return np.array(Image.open(image_path))
    nframes = len(exp_idx_seq)
    for i in range(nframes):
        yield read_frame(i)


def _load_frames_from_videos(src_dir, exp_idx_seq, motion_idx, round_size):
    def read_frame(i):
        cap_id = exp_idx_seq[i]
        video_path = os.path.join(src_dir, VIDEO_PATCH(cap_id))
        cap = cv2.VideoCapture(video_path)
        midx = (motion_idx + i) % (2 * round_size)
        midx = midx if midx < round_size else 2 * round_size - 1 - midx  # 往复循环
        cap.set(cv2.CAP_PROP_POS_FRAMES, midx)
        _, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    nframes = len(exp_idx_seq)
    for i in range(nframes):
        yield read_frame(i)


def video2video_patch(video_path, output_dir, patch_num: int = 1000):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    patch_size = nframes // patch_num
    pbar = tqdm()
    cnt, patch_idx = -1, -1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cnt += 1
        if cnt % patch_size == 0:
            if out is not None:
                out.release()
                pbar.update(1)
            patch_idx += 1
            out = cv2.VideoWriter(os.path.join(output_dir, VIDEO_PATCH(patch_idx)), fourcc, fps, (w, h))
        out.write(frame)
    out.release()
    cap.release()
