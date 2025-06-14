from deeptalk_logger import DeepTalkLogger
import queue
import traceback
import time
import numpy as np
import base64
import cv2
import threading
from asd_factory import ASDDetectorFactory


logger = DeepTalkLogger(__name__)

MAX_SILENCE_DURATION = 0.52


def proc(input_queue: queue.Queue, output_queue: queue.Queue, stop_event: threading.Event, config: dict):
    asd_model_type = config.get('asd_model_type', 'LR-ASD')
    video_fps = config.get('video_fps', 25)
    audio_sample_rate = config.get('audio_sample_rate', 16000)

    # factory = ASDDetectorFactory('LR-ASD', model_path='./LR_ASD/weight/finetuning_TalkSet.model')
    # factory = ASDDetectorFactory('TalkNet', model_path='./TalkNet_ASD/pretrain_AVA.model')
    # factory = ASDDetectorFactory('Light-ASD', model_path='./Light_ASD/weight/finetuning_TalkSet.model')
    factory = ASDDetectorFactory(asd_model_type, video_fps=video_fps, audio_sample_rate=audio_sample_rate)
    asd_detector = factory.asd_detector()

    start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_audio_time = time.perf_counter()
    try:
        while not stop_event.is_set():
            try:
                now = time.perf_counter()
                data_info = input_queue.get_nowait()  # 尝试不阻塞地从队列取出项
            except queue.Empty:  # 如果队列是空的，则等待一段时间
                time.sleep(0.01)  # 使用 asyncio 的 sleep
                continue
            if data_info is None:
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
                    gray_face_jpg_base64 = face_profile.get('image')
                    gray_face_jpg_buffer = base64.b64decode(gray_face_jpg_base64)
                    gray_face_jpg_nparray = np.frombuffer(gray_face_jpg_buffer, np.uint8)
                    gray_face_image = cv2.imdecode(gray_face_jpg_nparray, cv2.IMREAD_GRAYSCALE)
                    profile = {
                        "id": face_profile.get('id'),
                        "image": gray_face_image
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
                if vad_type == 1:
                    asd_detector.reset()
                asd_detector.append_audio(audio_chunk, create_time)
                if vad_type == 4:
                    start_evaluate_time = time.perf_counter()
                    all_face_scores = asd_detector.evaluate()
                    end_evaluate_time = time.perf_counter()
                    logger.info(f"ASD Detector cost time {(end_evaluate_time - start_evaluate_time):.3f} s")
                    score_info = {
                        "type": "score",
                        "create_time": create_time,
                        "scores": all_face_scores
                    }
                    output_queue.put_nowait(score_info)
                pass
    except Exception as e:
        logger.error(f"proc Error: {e}")
        traceback.print_exc()
    finally:
        logger.info(f"thread proc over.")
