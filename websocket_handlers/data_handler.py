from deeptalk_logger import DeepTalkLogger
import asyncio
import queue
import tornado.websocket
import tornado.ioloop
import time
import traceback
import json
import threading
from worker import proc
from concurrent.futures import ThreadPoolExecutor


logger = DeepTalkLogger(__name__)


class DataWebSocketHandler(tornado.websocket.WebSocketHandler):


    def initialize(self):
        self.stop_event = threading.Event()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.worker_thread = ThreadPoolExecutor(1, thread_name_prefix="worker")
        pass

    def open(self):
        logger.info(f"Data WebSocket opened")

        tornado.ioloop.IOLoop.current().spawn_callback(self.who_speak_run)

    # 用 async def 声明 who_speak_run 为一个异步函数
    async def who_speak_run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    score_info = self.output_queue.get_nowait()  # 尝试不阻塞地从队列取出项
                except queue.Empty:  # 如果队列是空的，则等待一段时间
                    await asyncio.sleep(0.01)  # 使用 asyncio 的 sleep
                    continue
                if score_info is None:
                    await asyncio.sleep(0.01)  # 遇到None则可能要退出或者做特别处理
                    continue
                self.write_message(score_info, binary=False)
                pass
        except Exception as e:
            logger.error(f"who_speak_run Error: {e}")
            traceback.print_exc()
        finally:
            logger.info(f"[DataWebSocketHandler]who_speak_run over.")

    def on_message(self, message):
        # logger.warning(f"DataWebSocketHandler======> {message}")
        json_message = json.loads(message)
        type = json_message.get('type')
        robot_id = json_message.get('robot_id')
        create_time = json_message.get('create_time')
        if type == 'init':
            config = json_message
            self.worker_thread.submit(proc, self.input_queue, self.stop_event, config)
            pass
        elif type == 'video':
            self.input_queue.put(json_message)
            pass
        elif type == 'audio':
            self.input_queue.put(json_message)
            pass
        pass


    def on_close(self):
        logger.info(f"Data WebSocket closed")
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.shutdown(wait=False, cancel_futures=True)
            self.worker_thread = False

    
    def check_origin(self, origin):
        # 允许任何源
        return True
