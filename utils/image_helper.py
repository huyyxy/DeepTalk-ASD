import cv2
import os
from datetime import datetime


# 全局变量，初始化序号
sequence_number = 1

def save_image_to_tmp(img):
    """
    将图像保存到当前目录下的 'tmp' 文件夹，以当前时间以及序号作为文件名。

    参数:
    - img: 要保存的图像
    
    返回值:
    - file_path: 保存的文件路径
    """
    global sequence_number  # 使用全局变量

    # 确保 'tmp' 目录存在
    os.makedirs('tmp', exist_ok=True)

    # 生成文件名，包含当前时间和序号
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{current_time}_{sequence_number}.jpg'
    file_path = os.path.join('tmp', file_name)

    # 保存图像为 JPG 文件
    is_saved = cv2.imwrite(file_path, img)

    # if is_saved:
    #     print(f'Image saved to {file_path}')
    # else:
    #     print('Failed to save the image.')

    # 递增序号
    sequence_number += 1

    return file_path

# 示例用法
# save_image_to_tmp(resized_mouth_img)
