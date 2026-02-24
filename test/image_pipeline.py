"""
脚本作用：
该脚本从系统默认摄像头捕获一张图像，将其转换为灰度JPEG图像，并使用base64进行编码，以演示通过网络传输的过程。
传输后，这张图像将从base64解码回JPEG格式，解码后显示在窗口中，以验证传输过程中图像数据的完整性。

脚本流程：
1. 脚本从设备的默认摄像头初始化视频捕获。
2. 有5秒的延迟以让视频流稳定。
3. 从视频流中捕获一帧并将其从BGR（OpenCV默认颜色空间）转换为灰度图像。
4. 将灰度图像编码成具有指定质量参数的JPEG格式。
5. 将JPEG图像数据编码为base64字符串以模拟数据打包进行传输。
6. 然后将base64字符串解码回JPEG字节。
7. 将JPEG字节读取回灰度图像数组。
8. 在窗口中显示恢复的灰度图像以进行视觉确认。

注意：
- 该脚本适用于演示基本的图像捕获、转换、编码/解码和显示过程。
- 脚本在内部模拟网络步骤，而无需实际传输，以使演示自成一体。
- 运行此脚本需要一个网络摄像头，并且OpenCV的`imshow`需要GUI环境。
"""
import time
import cv2
import base64
import numpy as np


if __name__ == "__main__":
    # 客户端流程
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头无法打开")
        exit(0)

    # 等待5秒,避免摄像头刚启动时较暗的视频帧
    time.sleep(5)

    ret, frame_bgr = cap.read()
    if not ret:
        print("视频帧无法获取")
        exit(0)
    
    print(f"frame_bgr ======> {frame_bgr.shape}")
    # 将BGR彩照转为灰度图
    gray_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    print(f"gray_image ======> {gray_image.shape}")

    # 将灰度图转为jpg
    quality = 70  # 你可以根据需要调整这个值，范围是0-100，100为最高质量
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, jpg_buffer = cv2.imencode('.jpg', gray_image, encode_param)
    if not retval:
        print("转为jpg失败")
        exit(0)
    print(f"jpg_buffer ======> {jpg_buffer.shape}")
    # 将jpg图像字节编码为base64编码格式的字符串
    jpg_base64 = base64.b64encode(jpg_buffer).decode('utf-8')
    # 假设经过互联网传输后，进入服务端
    # 服务端流程
    # 将base64编码格式的字符串解码为jpg图像字节
    jpg_buffer = base64.b64decode(jpg_base64)

    # 将jpg_buffer转换回灰度图像
    jpg_nparray = np.frombuffer(jpg_buffer, np.uint8)
    gray_image_restored = cv2.imdecode(jpg_nparray, cv2.IMREAD_GRAYSCALE)

    print(f"gray_image_restored ======> {gray_image_restored.shape}")

    # 可以在窗口中显示图像，检查是否正确
    cv2.imshow("Restored Gray Image", gray_image_restored)

    # 等待某个键被按下，然后关闭所有的窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
