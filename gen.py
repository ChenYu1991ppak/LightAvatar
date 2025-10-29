""" 基于搜索的实时数字人推流
    2核通用型CPU，实时推流(25fps)

    Date: 2025.02.21
"""
import os
import threading
import numpy as np
import queue
import time
from tqdm import tqdm
import soundfile as sf

from model import audio
from tools.common import TalkStatus
from tools.helper import _VideoHelper, _AudioHelper
from tools.stream import RTMPVideoStreamPusher, RTMPAudioStreamPusher
from tools.logger import logger


class AudioDrivenTalking(_VideoHelper):

    def __init__(self, src_dir: str):
        super().__init__(src_dir)
        self.wav_queue = queue.Queue()

    def __call__(self, wav: np.ndarray, output_folder: str):
        """ TalkingHead视频合成，即语音驱动视频人物表情
            :param wav: 目标音频数据
            :param output_folder: 输出目录
            :return: 生成视频完整路径
        """
        self.import_wav(wav, self.wav_queue)
        frames_out = list(self._search_speech_frames())
        result_path = self.write_mp4(frames=frames_out, output_folder=output_folder, fps=self.fps)
        temp_audio_path = os.path.join(output_folder, 'temp.wav')
        sf.write(temp_audio_path, wav, self.sr)
        video_path = self.add_audio(result_path, temp_audio_path, output_folder)
        os.remove(result_path)
        os.remove(temp_audio_path)
        return video_path

    def _search_speech_frames(self):
        _cnt = 0
        pbar = tqdm(total=self.wav_queue.qsize())
        while True:
            try:
                wav = self.wav_queue.get(timeout=2)
                embs = self.encode_wav(wav)
            except queue.Empty:
                pbar.close()
                break
            nframes = embs.shape[0]
            yield from self.search_frames(embs, _cnt)
            _cnt += nframes
            pbar.update(1)


class RealtimeAudioDrivenTalking(_VideoHelper):

    def __init__(self, src_dir: str, url: str = None):
        super().__init__(src_dir)
        self.wav_queue = queue.Queue()
        self._url = url
        self._start_ts = None
        self._status = TalkStatus.waiting
        self._is_alive = False

    @property
    def is_alive(self):
        return self._is_alive

    @property
    def status(self):
        return self._status.value

    def receive(self, wav: np.ndarray):
        self.import_wav(wav, self.wav_queue)

    def start(self):
        assert self._url is not None, "请设置推流URL后再启动推流。"
        if not self.is_alive:
            self._init_cache()
            self.streampusher = RTMPVideoStreamPusher(url=self._url, fps=self.fps, sr=self.sr)
            self.streampusher.start()
            self._start_stream_thread()
            self._start_ts = time.time()
            self._is_alive = True
        logger.info('Stream started.')

    def close(self):
        if self.is_alive:
            self._close_stream_thread()
            if hasattr(self, 'streampusher'):
                self.streampusher.close()
            self._is_alive = False
            self._start_ts = None
        self.wav_queue = queue.Queue()
        logger.info('Stream closed.')

    def stop_talking(self):
        self.wav_queue = queue.Queue()

    def run(self):
        st = time.time()
        while True:
            try:
                wav = self.wav_queue.get(block=False)
                embs = self.encode_wav(wav)
                frame_seq = list(self.search_frames(embs, motion_idx=self._nframes))
                self._status = TalkStatus.speaking
            except queue.Empty:
                wav = self._silence_wav
                frame_seq = list(self.get_silence_frames(nframes=self.min_nframes))
                self._status = TalkStatus.waiting

            nframes = len(frame_seq)
            wav_seq = self._split_wav(wav, max_nframes=nframes)
            yield frame_seq, wav_seq

            self._nframes += nframes
            # 限制生产速度，使其与时钟同步
            play_time = nframes / self.fps
            sleep_time = play_time - (time.time() - st)
            if sleep_time > 0:
                time.sleep(sleep_time)
            st = time.time()

    def _start_stream_thread(self):
        self._stream_event = threading.Event()
        self._stream_thread = threading.Thread(target=self._stream_task)
        self._stream_thread.start()

    def _close_stream_thread(self):
        self._stream_event.set()
        self._stream_thread.join()

    def _stream_task(self):
        it = iter(self.run())
        while not self._stream_event.is_set():
            frame_seq, wav_seq = next(it)
            self._push_into_stream(frame_seq, wav_seq)

    def _push_into_stream(self, frame_seq, wav_seq):
        for i in range(len(frame_seq)):
            frame = frame_seq[i][:, :, ::-1]  # to bgr
            wav_int16 = (wav_seq[i] * 32767).astype(np.int16)
            self.streampusher.push(frame, wav_int16)

    def _split_wav(self, wav, max_nframes: int = -1):
        _len_per_frame = int(self.sr * (1 / self.fps))
        if max_nframes >= 0:
            wav_splited = []
            for i in range(wav.shape[0] // _len_per_frame):
                if i >= max_nframes:
                    break
                wav_splited.append(wav[i * _len_per_frame:(i + 1) * _len_per_frame])
            return wav_splited
        else:
            return [wav]

    @staticmethod
    def _clear_queue(queue: queue.Queue):
        while queue.qsize() > 0:
            try:
                queue.get()
            except:
                pass

    def _record_status(self, interval=20):
        while True:
            time.sleep(interval)
            logger.info('Avg FPS: {}, Tar FPS: {}.'.format(
                self._nframes / (time.time() - self._start_ts),
                self.fps)
            )


class RealtimeAudioPlaying(_AudioHelper):

    def __init__(self, url: str):
        super().__init__()
        self.wav_queue = queue.Queue()

        self._url = url
        self._start_ts = None
        self._status = TalkStatus.waiting
        self._is_alive = False

        self._driving_event = threading.Event()

    @property
    def is_alive(self):
        return self._is_alive

    @property
    def status(self):
        return self._status.value

    def receive(self, wav: np.ndarray):
        self.import_wav(wav, self.wav_queue)

    def start(self):
        if not self.is_alive:
            self._init_cache()
            self.streampusher = RTMPAudioStreamPusher(url=self._url, sr=self.sr)
            self.streampusher.start()
            self._start_driving_thread()
            self._start_ts = time.time()
            self._is_alive = True
        logger.info('Stream started.')

    def close(self):
        if self.is_alive:
            self._close_driving_thread()
            if hasattr(self, 'streampusher'):
                self.streampusher.close()
            self._is_alive = False
            self._start_ts = None
        self.wav_queue = queue.Queue()
        logger.info('Stream closed.')

    def stop(self):
        self.wav_queue = queue.Queue()

    # TODO driving thread
    def _driving_task(self):
        st = time.time()
        while not self._driving_event.is_set():
            try:
                wav = self.wav_queue.get(block=False)
                self._status = TalkStatus.speaking
            except queue.Empty:
                wav = self._silence_wav
                self._status = TalkStatus.waiting

            self._push_into_stream(wav)
            # 限制生产速度，使其与时钟同步
            play_time = len(wav) / self.sr
            sleep_time = play_time - (time.time() - st)
            if sleep_time > 0:
                time.sleep(sleep_time)
            st = time.time()

    def _start_driving_thread(self):
        self._driving_event.clear()
        self._driving_thread = threading.Thread(target=self._driving_task)
        self._driving_thread.start()

    def _close_driving_thread(self):
        self._driving_event.set()
        self._driving_thread.join()

    def _push_into_stream(self, wav):
        """ 消费方调用
            生成延时: ...
        """
        wav = audio.wav2hex(wav)
        self.streampusher.push(wav)

