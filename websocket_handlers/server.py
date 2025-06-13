import os
import tornado.web
from websocket_handlers.data_handler import DataWebSocketHandler
import queue


def make_app():
    return tornado.web.Application([
        (r"/data", DataWebSocketHandler), # 用于接收音视频块
    ])


def start():
    port = int(os.environ.get('websocket_port', '7900'))
    app = make_app()
    app.listen(port)
    print(f"ASD webSocket server started on port {port}")
    tornado.ioloop.IOLoop.current().start()
