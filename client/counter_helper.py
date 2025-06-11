import time


class CallCounter:
    def __init__(self, message_template="一秒的平均调用次数：{:.3f}", print_interval=5):
        self.count = 0  # 初始化调用次数计数器
        self.start_time = time.perf_counter()  # 使用perf_counter获取更高精度的开始时间
        self.message_template = message_template  # 保存自定义的消息模板
        self.print_interval = print_interval  # 打印的间隔时间

    def call(self):
        self.count += 1  # 调用时计数器加一
        current_time = time.perf_counter()  # 获取当前时间
        # 判断自上次打印以来是否已超过5秒
        if current_time - self.start_time > self.print_interval:
            # 超过5秒时，根据提供的模板打印调用次数，并重置计数器和开始时间
            average = self.count / (current_time - self.start_time)
            print(self.message_template.format(average))
            self.count = 0
            self.start_time = current_time

class FaceInfoCounter:
    def __init__(self, print_interval=5):
        self.start_time = time.perf_counter()
        self.print_interval = print_interval
        self.total_face_count = 0
        self.total_msg_count = 0
        self.speak_count = 0
        self.strict_speak_count = 0
        self.speaking_duration_positive = 0
        self.max_speaking_duration = 0
        self.strict_speaking_duration_positive = 0
        self.max_strict_speaking_duration = 0
    
    def reset(self):
        self.total_face_count = 0
        self.total_msg_count = 0
        self.speak_count = 0
        self.strict_speak_count = 0
        self.speaking_duration_positive = 0
        self.max_speaking_duration = 0
        self.strict_speaking_duration_positive = 0
        self.max_strict_speaking_duration = 0
        self.start_time = time.perf_counter()

    def analyze_data(self, data):
        # print("[analyze_data]data ======>", data)
        if data is None:
            return
        self.total_face_count += len(data)
        self.total_msg_count += 1
        has_speak = False
        has_strict_speak = False
        for id, item in data.items():
            # print("[analyze_data]item ======>", item)
            speaking_duration = item.get('speak_duration', 0)
            if speaking_duration > 0:
                has_speak = True
                self.speaking_duration_positive += 1
                self.max_speaking_duration = max(self.max_speaking_duration, speaking_duration)

            strict_speaking_duration = item.get('strict_speak_duration', 0)
            if strict_speaking_duration > 0:
                has_strict_speak = True
                self.strict_speaking_duration_positive += 1
                self.max_strict_speaking_duration = max(self.max_strict_speaking_duration, strict_speaking_duration)
        if has_speak:
            self.speak_count += 1
        if has_strict_speak:
            self.strict_speak_count += 1

    def call(self, data):
        current_time = time.perf_counter()
        self.analyze_data(data)
        
        if current_time - self.start_time >= self.print_interval:
            elapsed_time = max(current_time - self.start_time, 1)  # 防止除以零
            average_calls_per_second = self.total_face_count / elapsed_time
            average_positive = self.speaking_duration_positive / elapsed_time
            average_last_positive = self.strict_speaking_duration_positive / elapsed_time
            
            print(f"一秒的平均接收人脸数: {average_calls_per_second:.3f}")
            print(f"平均每秒有人说话的脸帧数量: {average_positive:.3f}")
            print(f"平均每秒严格有人说话的脸帧数量: {average_last_positive:.3f}")
            print(f"有人说话的帧占比: {(self.speak_count/self.total_msg_count):.3f}")
            print(f"严格有人说话的帧占比: {(self.strict_speak_count/self.total_msg_count):.3f}")
            print(f"最大的说话时长: {self.max_speaking_duration:.3f}秒")
            print(f"最大的严格说话时长: {self.max_strict_speaking_duration:.3f}秒")
            
            # Reset counters and start_time for the next interval
            self.reset()
