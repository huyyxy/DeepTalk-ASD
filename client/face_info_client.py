import time
import os
import datetime
import logging
import threading
import cv2
import json
import base64
import traceback
from io import BytesIO
from inspireface_detect_manager import FaceDetectManager
from concurrent.futures import ThreadPoolExecutor
from cv2_capture import Cv2Capture
from counter_helper import CallCounter
import queue


# Setup basic logging
logging.basicConfig(level=logging.INFO)

ROBOT_ID = os.getenv('ROBOT_ID', '1')
FRAME_INTERVAL = 0  # seconds

face_profiles = {
    "face_count": 0,
    "profiles": {}
}

frame_counter = CallCounter("frame_num per sec======> {:.3f}")

class FaceInfoClient:
    """
    封装与Intel RealSense摄像头交互、人脸属性检测以及通过WebSocket发送数据的逻辑。
    """

    def __init__(self, output_queue:queue.Queue):
        """
        初始化摄像头并做人脸检测。

        Args:
            output_queue: output_queue。
        """
        self.face_detect_manager = FaceDetectManager({ "pyvision_emotion_model": None, "simplify_output": False })
        self.face_info_process_executor = ThreadPoolExecutor(max_workers=1)
        self.output_queue = output_queue
        
        self.last_focus_2_face_time = time.time()
        self.is_running = True
        self.first_zero = True
        self.send_face_info = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def stop(self):
        """Stop the client and WebSocket communication."""
        self.is_running = False
        self.face_info_process_executor.shutdown(wait=False)
        pass

    def detect_faces(self, color_frame):
        """
        检测frame中的人脸及其属性
        """
        color_image = color_frame
        face_profiles = self.face_detect_manager.detect(color_image)
        return face_profiles
    
    def capture_run(self, stop_event: threading.Event, camera_index: int=0):
        """
        捕获视频帧，检测人脸信息，并将其发送到WebSocket服务器。
        """
        time.sleep(1)
        # 打开特定index的摄像头
        cap = Cv2Capture(camera_index)
        cap.open()
        # 等待摄像头打开
        while cap.is_opened() is False:
            logging.info("等待摄像头开启...")
            if cap:
                cap.release()
                time.sleep(5)
                cap.open()
            time.sleep(0.1)

        try:
            while not stop_event.is_set():
                ret, frame_bgr = cap.read()
                # 确认视频帧被正确捕获
                if not ret:
                    logging.warning("Failed to capture video frame")
                    time.sleep(1.0)
                    continue

                # 由于使用OpenCV捕获的帧是BGR格式的，根据需要可能需要转换到RGB
                # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # 如果摄像头装反了，需要上下颠倒图像帧
                frame_bgr_flipped = cv2.flip(frame_bgr, -1)
                # frame_rgb_flipped = cv2.flip(frame_rgb, -1)
                
                frame_counter.call()

                self.face_info_process_executor.submit(self.detect_and_send_face, {"create_time": time.time(), "frame": frame_bgr})

                if FRAME_INTERVAL > 0.001:
                    time.sleep(FRAME_INTERVAL)
            logging.info("[capture_and_send] is not running")
        except Exception as e:
            logging.error(f"Error capturing and processing frame: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            print("====== capture_run finished ======")

    def detect_and_send_face(self, message):
        """
        在单线程池中处理，捕获视频帧，检测人脸信息，并将其发送到WebSocket服务器。
        """
        create_time = message["create_time"]
        if time.time() - create_time > 1.0:
            # 丢弃过期消息
            return
        
        color_frame: cv2.typing.MatLike = message["frame"]
        try:
            # logging.debug("enter detect_and_send_face ======>")
            # now = time.time()

            # 检测视频帧
            face_profiles = self.detect_faces(color_frame)
            # logging.info(f"face_profiles ======> {face_profiles}")

            face_list = []
            for id, profile in face_profiles.get('profiles', {}).items():
                face_image = profile.get('face_image')
                face_gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                # 设置 JPEG 图像的压缩质量
                quality = 70  # 你可以根据需要调整这个值，范围是0-100，100为最高质量
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                retval, face_jpg_buffer = cv2.imencode('.jpg', face_gray_image, encode_param)
                # 确保编码成功
                if not retval:
                    continue

                # 将图像字节编码为Base64
                face_jpg_base64 = base64.b64encode(face_jpg_buffer).decode('utf-8')

                face_list.append({
                    "id": id,
                    "image": face_jpg_base64
                })

            face_info = {
                "type": "video",
                "robot_id": ROBOT_ID,
                "create_time": create_time,
                "face_profiles": face_list,
            }
            # 每一帧的数据都发送给边缘服务器
            self.output_queue.put_nowait(face_info)
            # face_profiles_str = json.dumps(face_profiles, ensure_ascii=False)
        
        except Exception as e:
            logging.error(f"[FaceInfoClient]Error capturing and processing frame: {e}")
            traceback.print_exc()
