import os
import time
import traceback
from datetime import datetime
import asyncio
import tornado
from tornado.websocket import WebSocketHandler
from tornado.httpserver import HTTPServer

from gen import RealtimeAudioDrivenTalking
from tools.common import CODE, wscode, create_logger, SIGNAL_STOP_TALKING, SIGNAL_CHECK_STATUS, get_virtual_human_source_folder
from tools.tts import TTSBasedOnAzure

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
parser.add_argument('--azure_key', type=str, default='', help="Azure TTS key")
parser.add_argument('--azure_region', type=str, default='', help="Azure TTS region")
parser.add_argument('--voice_id', type=str, default='zh-CN-XiaoxiaoNeural', help="Azure TTS voice id")
parser.add_argument('--push_url', type=str, help="Retalking streaming push url")
parser.add_argument('--room_id', type=str, default='default_room', help="Retalking streaming room id")
parser.add_argument('--port', type=int, default=8889, help="Retalking streaming service port")

opt = parser.parse_args()

AVATAR_ID = opt.avatar_id
AZURE_KEY = opt.azure_key
AZURE_REGION = opt.azure_region
VOICE_ID = opt.voice_id
PUSH_URL = opt.push_url
ROOM_ID = opt.room_id
PORT = opt.port

print(f"Using avatar_id: {AVATAR_ID}\nvoice_id: {VOICE_ID}\npush_url\n{PUSH_URL}\nroom_id: {ROOM_ID}\nport: {PORT}")


class _InteractiveStreamingHandler(WebSocketHandler):
    max_connections_num = 0  # without limitation
    established_connections_num = 0
    logger = create_logger('InteractiveStream')

    def check_origin(self, origin: str) -> bool:
        return True

    async def open(self):
        start_time = time.time()
        try:
            push_url = PUSH_URL if PUSH_URL is not None else self.request.arguments['push_url'][0].decode()
            avatar_id = AVATAR_ID if AVATAR_ID is not None else self.request.arguments['avatar_id'][0].decode()
            voice_id = VOICE_ID if VOICE_ID is not None else self.request.arguments.get('voice_id', [b'zh-CN-XiaoxiaoNeural'])[0].decode()
            room_id = ROOM_ID if ROOM_ID is not None else self.request.arguments.get('room_id', [b'default_room'])[0].decode()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._stream_id = f'{room_id}_{timestamp}'  # 流的唯一标识
        except Exception as e:
            self.logger.error(f"[{self._stream_id}] 初始化视频流失败: {e}")
            self.logger.error(traceback.format_exc())
            self.close(code=wscode(CODE['SFAIL']), reason=str(e))

        self.logger.info(f"[{self._stream_id}] 推流接口被调用, 推流地址: {push_url}，数字人ID：{avatar_id}, 音色code：{voice_id}")

        if self.max_connections_num > 0 and self.established_connections_num >= self.max_connections_num:
            self.logger.warning(f"[{self._stream_id}] 超出最大连接数限制。当前连接数: {self.established_connections_num}")
            self.close(code=wscode(CODE['SFAIL_EXCEED']), reason='Exceeding the maximum connections.')
        else:
            try:
                self.established_connections_num += 1
                self.logger.info(f"[{self._stream_id}] 当前连接数增加至: {self.established_connections_num}")
                self._initialize_stream(avatar_id, voice_id, push_url)
                msg = {
                    'code': CODE['SUCC'],
                    'msg': 'Successfully established connection.',
                    'data': {'max_connections': self.max_connections_num,
                             'established_connections': self.established_connections_num,
                             'stream_id': self._stream_id,
                             }
                }
                await self.write_message(msg)
                self.logger.info(f"[{self._stream_id}] 数字人直播推流成功！连接到推流耗时：{time.time() - start_time:.3f}s")
            except Exception as e:
                print(traceback.format_exc())
                self.logger.error(f"[{self._stream_id}] 初始化推流失败: {e}")
                self.logger.error(traceback.format_exc())
                self.close(code=wscode(CODE['SFAIL_ALGO']), reason=str(e))

    async def on_message(self, text: str):
        start_time = time.time()
        self.logger.info(f"[{self._stream_id}] 接收消息: {text}")
        if text == SIGNAL_STOP_TALKING:
            self.stream_generator.stop()
            self.logger.info(f"[{self._stream_id}] 停止说话, 接收到暂停耗时：{time.time() - start_time:.3f}s")
        elif text == SIGNAL_CHECK_STATUS:
            msg = {
                'code': CODE['SUCC_RETURN'],
                'msg': 'Successfully returned.',
                'data': {'status': self.stream_generator.status}
            }
            await self.write_message(msg)
            self.logger.info(f"[{self._stream_id}] 说话状态: {msg}, 接收到查询耗时：{time.time() - start_time:.3f}s")
        else:
            try:
                total_bytes, total_duration, last_duration, lag_sec = 0, 0, 0, 0
                last_recieve_t = start_time
                async for wav in self.tts(text):
                    infer_cost_sec = time.time() - last_recieve_t
                    self.stream_generator.receive(wav)
                    last_recieve_t = time.time()
                    lag_sec += max(infer_cost_sec - last_duration, 0)
                    wav_size = len(wav)
                    total_bytes += wav_size
                    last_duration = wav_size / self.tts.sr
                    total_duration += last_duration
                self.logger.info(f"[{self._stream_id}]" \
                                 f"wav大小： {total_bytes} bytes， wav时长：{total_duration}，" \
                                 f"推理延迟：{lag_sec * 1000:3f}ms，推理耗时：{time.time() - start_time:.3f}s")
            except Exception as e:
                self.logger.error(f"[{self._stream_id}] 处理消息失败: {text}，错误: {e}")
                self.logger.error(traceback.format_exc())

    def on_close(self):
        try:
            self.stream_generator.close()
        except Exception as e:
            self.logger.warning(f"[{self._stream_id}] 关闭推流器时出错: {e}")
        self.established_connections_num -= 1
        self.logger.info(f"[{self._stream_id}] 连接断开。当前连接数: {self.established_connections_num}")

    def _initialize_stream(self, virtual_human_id, voice_id, lang, push_url):
        self.stream_generator = None
        self.tts = None


class RetalkingVideoStreamingHandler(_InteractiveStreamingHandler):
    max_connections_num = 1  # 放到2核pod上，最大支持1路
    established_connections_num = 0
    logger = create_logger('RetalkingStream')

    def _initialize_stream(self, virtual_human_id, voice_id, push_url):
        dataset_dir = get_virtual_human_source_folder(virtual_human_id)
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"[{self._stream_id}] 数字人{virtual_human_id}数据源不存在，加载失败！")
        else:
            self.logger.info(f"[{self._stream_id}] 数字人{virtual_human_id}数据加载完成，初始化TTS与推流器...")
            self.tts = TTSBasedOnAzure(azure_key=AZURE_KEY, azure_region=AZURE_REGION, voice_id=voice_id)
            self.logger.info(f"[{self._stream_id}] TTS后端：Azure")
            self.stream_generator = RealtimeAudioDrivenTalking(src_dir=dataset_dir, url=push_url)
            self.stream_generator.start()
            self.logger.info(f"[{self._stream_id}] 视频推流器启动成功!")


class RetalkingStreamingAPP(tornado.web.Application):
    _routes = [
        tornado.web.url(r"/api/algo/streaming/retalking_streaming", RetalkingVideoStreamingHandler),
    ]

    def __init__(self):
        super(RetalkingStreamingAPP, self).__init__(self._routes)


async def main(num_process=1):
    # 启动 RetalkingStreamingAPP 监听 S1_PORT
    retalking_app = RetalkingStreamingAPP()
    retalking_server = HTTPServer(retalking_app)
    retalking_server.bind(PORT)
    retalking_server.start(num_process)

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
