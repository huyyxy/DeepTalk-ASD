import time
import os
import datetime
import logging
import threading
import websocket
import json
import base64
import traceback
import queue
from concurrent.futures import ThreadPoolExecutor


# Setup basic logging
logging.basicConfig(level=logging.INFO)


ROBOT_ID = os.getenv('ROBOT_ID', '1')

class WebsocketClient:

    def __init__(self, ws_url, input_queue:queue.Queue, output_queue:queue.Queue, reconnect_delay=5):
        """
        初始化RealSense摄像头并设置WebSocket连接。

        Args:
            ws_url: WebSocket服务的URL。
            input_queue: 输入的消息。
            output_queue: 输出的消息。
            reconnect_delay: WebSocket重连的延迟时间，以秒为单位。
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ws_url = ws_url
        
        self.reconnect_delay = reconnect_delay
        self.ws = None
        self.is_running = True
        self.send_messages_executor = ThreadPoolExecutor(max_workers=1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ws:
            self.ws.close()
            self.ws = None

    def setup_websocket(self):
        """
        初始化WebSocket连接和回调事件。
        """
        if self.ws:
            self.ws.close()
            self.ws = None
        self.ws = websocket.WebSocketApp(self.ws_url, 
                                         on_open=self.on_open, 
                                         on_message=self.on_message, 
                                         on_error=self.on_error, 
                                         on_close=self.on_close, 
                                         on_ping=self.on_ping, 
                                         on_pong=self.on_pong, 
                                         )

    def run_forever(self):
        """
        保持运行状态，自动处理WebSocket的重连。
        """
        self.send_messages_executor.submit(self.process_message)
        while self.is_running:
            self.setup_websocket()
            try:
                self.ws.run_forever()
            except Exception as e:
                logging.error(f"[WebsocketClient]WebSocket run error: {e}")
            logging.info(f"[WebsocketClient]Reconnecting in {self.reconnect_delay} seconds...")
            time.sleep(self.reconnect_delay)
            num_threads = len(threading.enumerate())
            logging.info(f"[WebsocketClient]Number of active threads: {num_threads}")
        logging.info(f"[WebsocketClient]run_forever is not running")

    def stop(self):
        """Stop the client and WebSocket communication."""
        self.is_running = False
        if self.ws:
            self.ws.close()
            self.ws = None

    def on_open(self, ws:websocket.WebSocketApp):
        """Handle WebSocket opening and start sending frames."""
        logging.info(f"[WebsocketClient]WebSocket connection opened")
        # threading.Thread(target=self.capture_and_send_faces, daemon=True).start()
        init_message = {
            "robot_id":ROBOT_ID,
            "type":"init",
            "create_time":time.time(),
            "asd_model_type": "LR-ASD" # TalkNet Light-ASD LR-ASD
        }
        message_str = json.dumps(init_message, ensure_ascii=False)
        self.ws.send(message_str)

    def on_message(self, ws:websocket.WebSocketApp, message):
        logging.info(f"[WebsocketClient]Message received from server. message ======> {message}")
        data = json.loads(message)
        pass

    def on_error(self, ws:websocket.WebSocketApp, error):
        """Handle errors."""
        logging.error(f"[WebsocketClient]WebSocket error: {error}")

    def on_close(self, ws:websocket.WebSocketApp, close_status_code, close_msg):
        """Handle WebSocket closure."""
        logging.info(f"[WebsocketClient]WebSocket closed {close_status_code} {close_msg}")

    def on_ping(self, ws:websocket.WebSocketApp, message):
        logging.info(f"[WebsocketClient]WebSocket on_ping {message}")
        try:
            ws.sock.pong()
        except Exception as e:
            traceback.print_exc()
        pass

    def on_pong(self, ws:websocket.WebSocketApp, message):
        logging.info(f"[WebsocketClient]WebSocket on_pong")

    def process_message(self):
        """
        在单线程池中处理，捕获视频帧，检测人脸信息，并将其发送到WebSocket服务器。
        """
        while self.is_running:
            try:
                data_info = self.output_queue.get_nowait()  # 尝试不阻塞地从队列取出项
            except queue.Empty:  # 如果队列是空的，则等待一段时间
                time.sleep(0.1)
                continue
            if data_info is None:
                time.sleep(0.1)
                continue

            create_time = data_info["create_time"]
            if time.perf_counter() - create_time > 1.0:
                # 丢弃过期消息
                return
            
            try:
                # print(f"data_info ======> {data_info}")
                message_str = json.dumps(data_info, ensure_ascii=False)
                self.ws.send(message_str)
            except Exception as e:
                logging.error(f"[WebsocketClient]Error capturing and processing frame: {e}")
                traceback.print_exc()
