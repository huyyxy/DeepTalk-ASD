import os
import tornado.web
from websocket_handlers.data_handler import DataWebSocketHandler
import queue


def make_app(input_queue: queue.Queue, output_queue: queue.Queue):
    return tornado.web.Application([
        (r"/data", DataWebSocketHandler, dict(input_queue=input_queue, output_queue=output_queue)), # 用于接收音视频块
    ])


def start(input_queue: queue.Queue, output_queue: queue.Queue):
    port = int(os.environ.get('websocket_port', '7900'))
    app = make_app(input_queue, output_queue)
    app.listen(port)
    print(f"ASD webSocket server started on port {port}")
    tornado.ioloop.IOLoop.current().start()
