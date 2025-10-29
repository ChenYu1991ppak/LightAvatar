import os
import uuid
import ffmpeg
import subprocess
from queue import Queue
import threading


DEFAULT_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')


class _StreamPusher(object):

    def __init__(self,
                 url: str,
                 fps: int = 25,
                 sr: int = 16000,
                 prefix: str = DEFAULT_PREFIX,
                 channel: str = None,
                 ffmpeg_queue_size: int = 8):
        self.prefix = prefix
        self.channel = channel if channel else str(uuid.uuid4()).replace('-', '_')
        subprocess.run(f"mkdir -p {self.prefix}", shell=True, check=True)

        self.fps = fps
        self.sr = sr
        self.ffmpeg_queue_size = ffmpeg_queue_size
        self._size = None  # H, W
        self._url = url
        self._is_alive = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    def push(self, *args, **kwargs):
        pass

    def start(self):
        self._on_start()

    def _on_start(self):
        pass

    def close(self):
        self._on_close()

    def _on_close(self):
        pass

    def _run_stream_process(self):
        pass

    @staticmethod
    def _writing_process(pipe, queue):
        with open(pipe, 'wb') as fd:
            while True:
                try:
                    b = queue.get()
                    if b is None:
                        break
                    fd.write(b)
                    fd.flush()
                except BrokenPipeError:
                    break

    @staticmethod
    def clear_queue(queue: Queue):
        while not queue.empty():
            queue.get()


class RTMPVideoStreamPusher(_StreamPusher):
    def __init__(self,
                 url: str,
                 fps: int = 25,
                 sr: int = 16000,
                 prefix: str = DEFAULT_PREFIX,
                 channel: str = None,):
        assert url.startswith('rtmp://')
        super(RTMPVideoStreamPusher, self).__init__(url=url, fps=fps, sr=sr, prefix=prefix, channel=channel)
        self.video_pipe = os.path.join(self.prefix, '_'.join([self.channel, 'video']))
        self.audio_pipe = os.path.join(self.prefix, '_'.join([self.channel, 'audio']))
        print(self.video_pipe, self.audio_pipe)
        self.frame_queue, self.speech_queue = Queue(), Queue()
        self.push_process = None
        self.video_thread, self.audio_thread = None, None

    def _run_stream_process(self):
        """ 构建并启动基于ffmpeg的推流线程 """
        assert self._size is not None
        video_in = ffmpeg.input(
            self.video_pipe,
            thread_queue_size=str(self.ffmpeg_queue_size),
            format='rawvideo',
            pix_fmt='bgr24',
            r=str(self.fps),
            s="{}x{}".format(self._size[1], self._size[0])
        )
        audio_in = ffmpeg.input(
            self.audio_pipe,
            thread_queue_size=str(self.ffmpeg_queue_size),
            format='s16le',
            acodec='pcm_s16le',
            ar=str(self.sr)
        )
        push_process = (
            ffmpeg.output(
                video_in,
                audio_in,
                self.url,
                # loglevel='quiet',
                vcodec='libx264',
                acodec='aac',
                threads='1',
                ac='2',
                ar=str(self.sr),
                r=str(self.fps),
                # ab='320k',
                video_bitrate='1500k',
                # preset='ultrafast',
                preset='veryfast',  # 避免 ultrafast 的过度优化
                tune='zerolatency',
                # shortest=True,
                g='25',
                f='flv',
                # 关键修正参数：
                pix_fmt='yuv420p',  # 强制输出为 yuv420p
                x264opts='keyint=4:min-keyint=4:no-scenecut',  # 固定 GOP 结构
                profile='main',  # 兼容 Chrome 的 Profile
                level='3.1',  # 限制 Level 为 3.1
                flags='+global_header'  # 确保 FLV 封装包含头信息
            ).overwrite_output()
            .run_async(cmd=["ffmpeg", "-re"], pipe_stdin=True)
        )  # 推流
        return push_process

    def push(self, frame, speech):
        """ 此处接收一帧
        :param frame: BGR image
        :param speech:
        :return:
        """
        assert len(speech) == (self.sr // self.fps) and (self.sr % self.fps) == 0
        if not self.is_alive:
            self._size = frame.shape[:2]
            self.start()
        assert frame.shape[:2] == self._size, 'frame shape must be consistence.'
        self.frame_queue.put(frame.tobytes())
        self.speech_queue.put(speech.tostring())

    def _on_start(self):
        if not self.is_alive and self._size is not None:
            os.mkfifo(self.video_pipe)
            os.mkfifo(self.audio_pipe)
            self.push_process = self._run_stream_process()
            self.video_thread = threading.Thread(target=self._writing_process, args=(self.video_pipe, self.frame_queue))
            self.audio_thread = threading.Thread(target=self._writing_process, args=(self.audio_pipe, self.speech_queue))
            self.video_thread.start()
            self.audio_thread.start()
            self._is_alive = True

    def _on_close(self):
        if self.is_alive:
            self.push_process.terminate()
            os.remove(self.video_pipe)
            os.remove(self.audio_pipe)
            self.frame_queue, self.speech_queue = Queue(), Queue()
            self._is_alive = False


class RTMPAudioStreamPusher(_StreamPusher):

    def __init__(self,
                 url: str,
                 sr: int = 16000,
                 prefix: str = DEFAULT_PREFIX,
                 channel: str = None,):
        assert url.startswith('rtmp://')
        super(RTMPAudioStreamPusher, self).__init__(url=url, sr=sr, prefix=prefix, channel=channel)
        self.audio_pipe = os.path.join(self.prefix, '_'.join([self.channel, 'audio']))
        self.speech_queue = Queue()
        self.push_process = None
        self.audio_thread = None

    def _run_stream_process(self):
        """ 构建并启动基于ffmpeg的推流线程 """
        audio_in = ffmpeg.input(
            self.audio_pipe,
            thread_queue_size=str(self.ffmpeg_queue_size),
            format='s16le',
            acodec='pcm_s16le',
            ar=str(self.sr)
        )
        push_process = (
            ffmpeg.output(
                audio_in,
                self.url,
                acodec='aac',
                threads='1',
                ac='2',
                ar=str(self.sr),
                preset='ultrafast',  # 避免 ultrafast 的过度优化
                tune='zerolatency',
                probesize='32',  # 减小探测大小
                analyzeduration='0',  # 减少分析时间
                fflags='nobuffer',  # 不缓冲任何帧
                flags='low_delay',  # 启用低延迟选项
                flush_packets='1',  # 立即刷新数据包
                max_delay='0',  # 减少延迟
                flvflags='no_duration_filesize',  # 减少元数据开销
                f='flv',
                live='1',  # 优化用于直播
            ).overwrite_output()
            .run_async(cmd=["ffmpeg", "-re"], pipe_stdin=True)
        )  # 推流
        return push_process

    def push(self, speech):
        """ 此处接收一帧
        :param frame: BGR image
        :param speech:
        :return:
        """
        if not self.is_alive:
            self.start()
        self.speech_queue.put(speech.tostring())

    def _on_start(self):
        if not self.is_alive:
            os.mkfifo(self.audio_pipe)
            self.push_process = self._run_stream_process()
            self.audio_thread = threading.Thread(target=self._writing_process, args=(self.audio_pipe, self.speech_queue))
            self.audio_thread.start()
            self._is_alive = True

    def _on_close(self):
        if self.is_alive:
            self.push_process.terminate()
            os.remove(self.audio_pipe)
            self.speech_queue = Queue()
            self._is_alive = False
