import os
import time
import logging
import pyaudio
import base64
import traceback
import numpy as np
import queue
import threading
from enum import Enum, unique
from scipy.signal import resample
from concurrent.futures import ThreadPoolExecutor
from pyvadfactory import Factory as VadFactory


# Setup basic logging
logging.basicConfig(level=logging.INFO)


MIC_DEVICE_NAME = os.getenv('MIC_DEVICE_NAME', 'MacBook Pro麦克风')

# 麦克风配置
INPUT_FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
INPUT_CHANNELS = 1  # Mono audio
INPUT_RATE = 16000  # Sample rate
INPUT_CHUNK_SIZE = int(INPUT_RATE * 0.03)  # Block size
# INPUT_FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
# INPUT_CHANNELS = 2  # Mono audio
# INPUT_RATE = 48000  # Sample rate
# INPUT_CHUNK_SIZE = int(INPUT_RATE * 1)  # Block size

OUTPUT_RATE = 16000


def find_sound_device_index(device_name:str):
    p = pyaudio.PyAudio()
    device_index = None  # 默认的设备索引为空
    # 遍历可用的音频输入设备
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if device_name in dev['name']:
            print(f"Found Device '{dev['name']}' at Index {i}")
            device_index = i
            break
    
    p.terminate()
    return device_index

def convert_audio(input_data, input_rate=INPUT_RATE, output_rate=OUTPUT_RATE, input_channels=INPUT_CHANNELS):
    """
    Convert audio from one sample rate to another, and from stereo to mono.

    Parameters:
    - input_data: PCM data as bytes.
    - input_rate: Sample rate of the input data. Default is 48kHz.
    - output_rate: Desired sample rate of the output data. Default is 16kHz.
    - input_channels: Number of channels in the input data. Default is 2 (stereo).

    Returns:
    - output_data: Converted audio as bytes, in mono and at the desired sample rate.
    """
    # Convert input bytes data to numpy array
    pcm_np = np.frombuffer(input_data, dtype=np.int16)

    # Reshape the array to separate channels and then take the average to convert to mono
    if input_channels > 1:
        pcm_np = pcm_np.reshape(-1, input_channels).mean(axis=1)

    # Calculate the number of output samples
    num_output_samples = int(len(pcm_np) * output_rate / input_rate)

    # Resample from input_rate to output_rate
    resampled_np = resample(pcm_np, num_output_samples)

    # Convert numpy array back to bytes
    output_data = resampled_np.astype(np.int16).tobytes()

    return output_data

@unique
class VADType(Enum):
    NON_SPEECH = -1  # 非人声
    VAD_START = 1  # VAD开始
    VAD_CONTINUE = 2  # VAD继续
    VAD_SILENCE = 3  # VAD中的无人说话的持续帧
    VAD_END_NORMAL = 4  # VAD正常结束
    VAD_END_LOW_VOLUME = 5  # VAD结束时判断为低音量，丢弃
    VAD_DISCARD_SHORT = 6  # 特别短的VAD，丢弃

    def __str__(self):
        return self.name

class SpeechInfoClient:
    """
    封装与Intel RealSense摄像头交互、人脸属性检测以及通过WebSocket发送数据的逻辑。
    """

    def __init__(self, output_queue:queue.Queue):
        """
        初始化麦克风并进行人声检测。

        Args:
            output_queue: output_queue。
        """
        vad_factory = VadFactory('pyvad', sub_vad_type='webrtcvad', threshold_db=5, mode=3)
        self.vad = vad_factory.create_vad()
        self.speech_info_process_executor = ThreadPoolExecutor(max_workers=1)
        self.output_queue = output_queue
        self.is_running = True
        self.speech_start_time = 0
        self.no_speech_count = 0
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def stop(self):
        """Stop the client and WebSocket communication."""
        self.is_running = False
        self.speech_info_process_executor.shutdown(wait=False)

    def _is_speech(self, buf, sample_rate):
        return self.vad.is_speech(buf, sample_rate)

    def mic_run(self, stop_event: threading.Event):
        device_name = MIC_DEVICE_NAME
        device_index = find_sound_device_index(device_name)
        if device_index is None:
            print("======> 声卡设备接触不良 <======")
            return
        
        try:
            p = pyaudio.PyAudio()
            mic_stream = p.open(format=INPUT_FORMAT,
                channels=INPUT_CHANNELS,
                rate=INPUT_RATE,
                input=True,
                frames_per_buffer=INPUT_CHUNK_SIZE,
                input_device_index=device_index,
                )
        except Exception as e:
            print("create pyaudio error =======>", e)
            return

        print("Recording Mic...")

        try:
            while not stop_event.is_set():
                if mic_stream.is_active():
                    org_pcm_data = mic_stream.read(INPUT_CHUNK_SIZE)
                    self.speech_info_process_executor.submit(self.process_and_send_speech, {"create_time": time.perf_counter(), "data": org_pcm_data})
                else:
                    print("=== mic_stream is not active ===")
        except Exception as e:
            print(f"mic_run Error: {e}")
            traceback.print_exc()
        finally:
            if mic_stream is not None and mic_stream.is_active():
                mic_stream.stop_stream()
                mic_stream.close()
            p.terminate()
            print("====== mic_run finished ======")

    def process_and_send_speech(self, message):
        """
        在单线程池中处理，捕获视频帧，检测人脸信息，并将其发送到WebSocket服务器。
        """
        create_time = message["create_time"]
        if time.perf_counter() - create_time > 1.0:
            # 丢弃过期消息
            return
        
        org_pcm_data = message["data"]
        try:
            # logging.debug("enter process_and_send_speech ======>")

            if INPUT_RATE != OUTPUT_RATE or INPUT_CHANNELS != 1:
                # 检测视频帧
                dst_pcm_data = convert_audio(org_pcm_data, INPUT_RATE, OUTPUT_RATE, INPUT_CHANNELS)
            else:
                dst_pcm_data = org_pcm_data

            vad_type = VADType.NON_SPEECH
            speech_info = {
                "type": "audio",
                "robot_id": "1",
                "create_time": create_time,
                "audio_chunk": base64.b64encode(org_pcm_data).decode('utf-8'),
            }

            is_speech = self._is_speech(dst_pcm_data, OUTPUT_RATE)
            if is_speech:
                if self.speech_start_time < 0.001 and self.speech_start_time > -0.001:
                    self.speech_start_time = time.perf_counter()
                    vad_type = VADType.VAD_START
                else:
                    vad_type = VADType.VAD_CONTINUE
                self.no_speech_count = 0
            else:
                if self.speech_start_time > 0.001 and self.no_speech_count < 60:
                    vad_type = VADType.VAD_SILENCE
                    self.no_speech_count += 1
                else:
                    if time.perf_counter() - self.speech_start_time > 0.52:
                        vad_type = VADType.VAD_END_NORMAL
                    elif self.speech_start_time > 0.001:
                        vad_type = VADType.VAD_DISCARD_SHORT
                    else:
                        vad_type = VADType.NON_SPEECH
                    self.speech_start_time = 0.0
            
            speech_info["vad_type"] = vad_type.value
            self.output_queue.put_nowait(speech_info)
        except Exception as e:
            logging.error(f"[SpeechInfoClient]Error process_and_send_speech: {e}")
            traceback.print_exc()
