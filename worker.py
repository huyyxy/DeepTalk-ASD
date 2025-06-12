from deeptalk_logger import DeepTalkLogger
import queue
import traceback
import time
import numpy as np
import base64
import cv2
from asd_factory import ASDDetectorFactory


logger = DeepTalkLogger(__name__)

MAX_SILENCE_DURATION = 0.52


def proc(input_queue: queue.Queue):
    # factory = ASDDetectorFactory('LR-ASD', model_path='/home/william/Workspace/DeepTalk-ASD/LR_ASD/weight/finetuning_TalkSet.model')
    factory = ASDDetectorFactory('TalkNet', model_path='/home/william/Workspace/DeepTalk-ASD/TalkNet_ASD/pretrain_AVA.model')
    asd_detector = factory.asd_detector()

    start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_audio_time = time.perf_counter()
    try:
        while True:
            try:
                now = time.perf_counter()
                data_info = input_queue.get_nowait()  # 尝试不阻塞地从队列取出项
            except queue.Empty:  # 如果队列是空的，则等待一段时间
                # if now - last_audio_time > MAX_SILENCE_DURATION:
                #     # 
                #     asd_detector.evaluate()
                #     # asd_detector.reset()
                #     last_audio_time = now
                #     pass
                time.sleep(0.01)  # 使用 asyncio 的 sleep
                continue
            if data_info is None:
                # if now - last_audio_time > MAX_SILENCE_DURATION:
                #     # 
                #     asd_detector.evaluate()
                #     # asd_detector.reset()
                #     last_audio_time = now
                #     pass
                time.sleep(0.01)  # 遇到None则可能要退出或者做特别处理
                continue
            
            create_time = data_info.get('create_time')
            if create_time - last_time > 1.0:
                start_time = create_time
            last_time = create_time

            type = data_info.get('type')
            if type == 'video':
                create_time = data_info.get('create_time')
                face_profiles = data_info.get('face_profiles')
                profiles = []
                for face_profile in face_profiles:
                    face_image_base64 = face_profile.get('image')
                    face_image_bytes = base64.b64decode(face_image_base64)
                    # 将字节流转换为一维 numpy 数组
                    face_image_array = np.frombuffer(face_image_bytes, dtype=np.uint8)
                    # 使用 cv2.imdecode 解码为 OpenCV 图像
                    face_image = cv2.imdecode(face_image_array, cv2.IMREAD_COLOR)
                    profile = {
                        "id": face_profile.get('id'),
                        "image": face_image
                    }
                    profiles.append(profile)
                asd_detector.append_video(profiles, create_time)
                pass
            elif type == 'audio':
                # print(f"audio data_info ======> {data_info}")
                create_time = data_info.get('create_time')
                audio_chunk_base64 = data_info.get('audio_chunk')
                vad_type = data_info.get('vad_type', -1)
                audio_bytes = base64.b64decode(audio_chunk_base64)
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                asd_detector.append_audio(audio_chunk, create_time)
                last_audio_time = create_time
                if vad_type == 4:
                    asd_detector.evaluate()
                pass

            # if now - last_audio_time > MAX_SILENCE_DURATION:
            #     # 
            #     asd_detector.evaluate()
            #     # asd_detector.reset()
            #     last_audio_time = create_time
            #     pass
            

    except Exception as e:
        logger.error(f"proc Error: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"[DataWebSocketHandler]proc over.")











