import os
import traceback
import logging
import threading
import signal
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from face_info_client import FaceInfoClient
from speech_info_client import SpeechInfoClient, find_sound_device_index
from websocket_client import WebsocketClient
from cv2_capture import Cv2Capture
import time

# Configurations
ASD_WS_URL = os.getenv('ASD_WS_URL', 'ws://192.168.20.187:7900/data')

CAMERA_INDEX = 0
MIC_DEVICE_NAME = os.getenv('MIC_DEVICE_NAME', 'MacBook Pro麦克风')
FRAME_INTERVAL = 0  # seconds
RECONNECT_DELAY = 5  # seconds

face_info_process_executor = ThreadPoolExecutor(max_workers=1)
face_info_main_executor = ThreadPoolExecutor(max_workers=1)
speech_main_executor = ThreadPoolExecutor(max_workers=1)
# Setup basic logging
logging.basicConfig(level=logging.INFO)


# 处理信号的函数
def signal_handler(signum, frame, stop_event, face_info_client:FaceInfoClient, speech_client:SpeechInfoClient, websocket_client:WebsocketClient):
    logging.warning(f"Received signal {signum}, stopping threads...")
    stop_event.set()
    face_info_client.stop()
    speech_client.stop()
    websocket_client.stop()

if __name__ == "__main__":
    logging.info("\r\n\r\n====== start main ======\r\n\r\n")
    camera_ready = False
    cap = Cv2Capture(CAMERA_INDEX)
    while camera_ready is not True:
        camera_ready = cap.check_camera_ready()
        if camera_ready is not True:
            logging.info("暂时找不到摄像头设备，5秒钟后再试...")
            time.sleep(5)

    device_index = None
    while device_index is None:
        device_name = MIC_DEVICE_NAME
        device_index = find_sound_device_index(device_name)
        if device_index is None:
            print(f"暂时找不到拾音设备[{device_name}]，5秒钟后再试...")
            time.sleep(5)
    
    # 主线程处理websocket


    # 创建事件对象以便主线程与工作线程间同步
    stop_event = threading.Event()
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    websocket_client = WebsocketClient(ASD_WS_URL, input_queue, output_queue, RECONNECT_DELAY)
    face_info_client = FaceInfoClient(output_queue)
    speech_client = SpeechInfoClient(output_queue)
    # 定义要捕获的信号及其对应的处理器
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: signal_handler(signum, frame, stop_event, face_info_client, speech_client, websocket_client))

    face_info_main_executor.submit(face_info_client.capture_run, stop_event)
    speech_main_executor.submit(speech_client.mic_run, stop_event)

    try:
        websocket_client.run_forever()
    except Exception as e:
        traceback.print_exc()


