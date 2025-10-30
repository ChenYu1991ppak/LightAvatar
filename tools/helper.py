# TODO move to tools/helper.py
import cv2
import numpy as np
import os
import platform
import subprocess
import queue

from model.module import AudioEnCoder
from model.pitch import extract_f0_from_wav_and_mel
from model import audio
from model.hparams import hparams as hp
from tools.search import SearchFramesByAudioSegment
from tools.common import DEFAULT_CONFIG_FILE, load_model_paths


class _BaseHelper(object):
    """ 包含生成视频通用methods """

    def __init__(self):
        self.batch_size = 8

        # 音视频超参数
        self.fps = hp.fps
        self.sr = hp.sample_rate
        self.hop_size = hp.hop_size
        self.mel_step_size, self.mel_idx_multiplier, i = 16, 80. / self.fps, 0
        self.wav_size = self.mel_step_size * self.hop_size  # 推理一帧所需的时序信号长度

        self.min_sec = 0.24  # 可被处理的最小音频时长；降低该数值会增加计算开销，但会减少延迟
        self.min_nframes = int(self.min_sec * self.fps)
        self.ns = int(self.min_sec * self.sr)
        self.ns_padded = int(self.ns + self.wav_size)
        self.max_enc_sec = 1.  # 音频切段所占播放时长上限（对于实时输入的音频将切分成若干片段）
        self.ns_enc_padded = int(self.max_enc_sec * self.sr + self.wav_size)
        self._silence_wav = self._make_empty_wav(self.min_sec)  # 空白音频用于填充实时流，模拟说话间隙

        self._init_cache()

    def _init_cache(self):
        self.wav_cache = np.zeros(shape=(0,), dtype=np.float32)
        self._nframes = 0  # 总播放帧数

    def _make_empty_wav(self, sec: float):
        wav_size = int(sec * hp.sample_rate) + self.mel_step_size * self.hop_size
        wav = np.zeros((wav_size, ), dtype=np.float32)
        return wav

    @staticmethod
    def write_mp4(frames, output_folder: str, output_name: str = 'result', fps: float = 25.):
        os.makedirs(output_folder, exist_ok=True)
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        gen_path = os.path.join(output_folder, 'gen.mp4')
        result_path = os.path.join(output_folder, '{}.mp4'.format(output_name))
        writer = cv2.VideoWriter(gen_path, fourcc, fps, (W, H), True)
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        command = 'ffmpeg -loglevel error -y -i {} -strict -2 -q:v 1 {}'.format(
            gen_path, result_path)
        subprocess.call(command, shell=platform.system() != 'Windows')
        os.remove(gen_path)
        return result_path

    @staticmethod
    def add_audio(gen_path, audio_path, output_folder):
        out_file = 'output.mov' if gen_path.endswith('.mov') else 'output.mp4'
        out_path = os.path.join(output_folder, out_file)
        command = 'ffmpeg -loglevel error -y -i {} -i {} -c:v copy -c:a aac {}'.format(
            audio_path, gen_path, out_path)
        subprocess.call(command, shell=platform.system() != 'Windows')
        return out_path


class _VideoHelper(_BaseHelper):

    def __init__(self, src_dir: str):
        super().__init__()
        cfg_path = DEFAULT_CONFIG_FILE
        _, model_paths = load_model_paths(cfg_path)
        self.model_paths = model_paths

        lang = os.path.basename(src_dir).split('_')[1].upper()
        self.encoder = AudioEnCoder(onnx_path=self.model_paths['audio_encoder'][lang])
        self.search_frames = SearchFramesByAudioSegment(src_dir=src_dir, cluster_path=self.model_paths['clusters'][lang])

    def import_wav(self, wav, q: queue.Queue):
        self.wav_cache = np.concatenate((self.wav_cache, wav), axis=0)
        while True:
            if self.wav_cache.shape[0] > self.ns_padded:
                # 将音频切分成最小的推理单元，切分粒度越小延迟越低且计算代价越高
                wav = self.wav_cache[: self.ns_enc_padded]
                wav_len = wav.shape[0]
                if wav_len >= self.ns_padded:
                    q.put(wav)
                    self.wav_cache = self.wav_cache[wav_len - self.wav_size:]
                else:
                    break
            else:
                break

    def get_silence_frames(self, nframes):
        if not hasattr(self, 'silence_exp_idx'):
            embs = self.encode_wav(self._silence_wav)
            silence_exp_idx = self.search_frames.find_exp_idx(embs)[0]
            self.silence_exp_idx = silence_exp_idx
        exp_idx_seq = [self.silence_exp_idx for _ in range(nframes)]
        yield from self.search_frames.frame_loader(exp_idx_seq=exp_idx_seq, motion_idx=self._nframes)

    def encode_wav(self, wav):
        max_num = int((wav.shape[0] - self.wav_size) / self.mel_idx_multiplier / self.hop_size)
        mel = audio.melspectrogram(wav)
        mel_len = mel.shape[1]
        f0 = extract_f0_from_wav_and_mel(wav, mel.T, hop_size=self.hop_size, audio_sample_rate=self.sr)
        mel_f0_seq, i = [], 0
        while True:
            start_idx = int(i * self.mel_idx_multiplier)
            if start_idx + self.mel_step_size > mel_len or start_idx >= mel_len - 1:
                break
            mel_chunk = mel[:, start_idx: start_idx + self.mel_step_size]
            f0_chunk = f0[start_idx: start_idx + self.mel_step_size]
            mel_f0_seq.append((mel_chunk, f0_chunk))
            i += 1
            if i >= max_num:
                break
        return self._encode(mel_f0_seq) if i > 0 else None

    def _encode(self, mel_f0_seq):
        embs = []
        mel_chunks, f0_chunks = zip(*mel_f0_seq)
        for bmel_chunks, bf0_chunks in self._yield_embedding_batch(
                mel_chunks, f0_chunks, self.batch_size):
            bmels = np.reshape(bmel_chunks, [len(bmel_chunks), bmel_chunks.shape[1], bmel_chunks.shape[2], 1])
            bembs = self.encoder(bmels, bf0_chunks)
            embs.extend(bembs)
        embs = np.concatenate(embs, axis=0).reshape(-1, 512)
        return embs

    @staticmethod
    def _yield_embedding_batch(mel_chunks, f0_chunks, batch_size=8):
        nframes = len(mel_chunks)
        for idx in range(nframes // batch_size + 1):
            if idx * batch_size < nframes:
                bmel_chunks = np.asarray(mel_chunks[idx * batch_size: (idx + 1) * batch_size])
                bf0_chunks = np.asarray(f0_chunks[idx * batch_size: (idx + 1) * batch_size])
                yield bmel_chunks, bf0_chunks
            else:
                break


class _AudioHelper(_BaseHelper):

    def __init__(self):
        super().__init__()
        self._init_cache()

    def import_wav(self, wav, q: queue.Queue):
        """
        :param wav:
        :param q:
        :return:
        """
        self.wav_cache = np.concatenate((self.wav_cache, wav), axis=0)
        while True:
            if self.wav_cache.shape[0] > self.ns:
                wav = self.wav_cache[: self.ns]
                q.put(wav)
                self.wav_cache = self.wav_cache[self.ns:]
            else:
                break
