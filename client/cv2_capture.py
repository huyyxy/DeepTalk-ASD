import logging
import cv2


# Setup basic logging
logging.basicConfig(level=logging.INFO)

RECONNECT_DELAY = 5  # seconds
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 15
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 25


class Cv2Capture:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        pass
    
    def check_camera_ready(self):
        """
        检查指定索引的摄像头设备是否准备就绪。

        参数:
        camera_index (int): 摄像头索引号，默认为0。

        返回:
        bool: 如果摄像头准备就绪且能成功打开，则返回True，否则返回False。
        """
        cap = None
        try:
            # 尝试使用指定索引创建视频捕获对象
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                logging.warning("摄像头无法打开！")
                return False

            # 尝试从摄像头读取一帧
            ret, frame = cap.read()
            if not ret:
                logging.warning("无法从摄像头读取数据！")
                return False

            # 成功读取数据，返回成功
            logging.info("摄像头已准备就绪！")
            return True
        except Exception as e:
            logging.error(f"检测摄像头时发生异常：{e}")
            return False
        finally:
            # 释放摄像头资源
            if cap:
                cap.release()

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            # GR2目前的摄像头，需要加这一句
            # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # 设置摄像头参数，如果需要的话
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
