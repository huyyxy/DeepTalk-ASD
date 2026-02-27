import math
import numpy as np


def convert_to_box(face_rectangle):
    """
    将脸部矩形定义为 (x, y, width, height) 格式转换为
    边界框定义为 (x1, y1, x2, y2) 格式。

    :param face_rectangle: 字典，包含 "x", "y", "width" 和 "height" 键。
    :return: 表示边界框的列表 [x1, y1, x2, y2]。
    """
    x = face_rectangle["x"]
    y = face_rectangle["y"]
    width = face_rectangle["width"]
    height = face_rectangle["height"]
    return [x, y, x + width, y + height]


def _center_of_box(box):
    """计算框的中心点."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def _distance_between_centers(center1, center2):
    """计算两个中心点之间的距离."""
    x1, y1 = center1
    x2, y2 = center2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _are_boxes_close(box1, box2, max_distance):
    """检查两个框的中心点是否足够接近."""
    center1 = _center_of_box(box1)
    center2 = _center_of_box(box2)
    return _distance_between_centers(center1, center2) <= max_distance


def _area_of_box(box):
    """计算框的面积."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def _are_areas_similar(area1, area2, threshold):
    """检查两个面积是否在给定的阈值内相似."""
    size_difference = abs(area1 - area2) / max(area1, area2)
    return size_difference <= threshold


def _are_angles_similar(angles1, angles2, threshold):
    """检查两个角度列表是否在给定的阈值内相似."""
    return all(abs(a1 - a2) < threshold for a1, a2 in zip(angles1, angles2))


def is_same_face(box1, box2, angles1, angles2,
                 distance_threshold=30,
                 size_difference_threshold=0.5,
                 angle_difference_threshold=15):
    """
    基于空间和角度相似性判断两个脸是否相同。

    :param box1: 第一个脸的边界框 [x1, y1, x2, y2]。
    :param box2: 第二个脸的边界框 [x1, y1, x2, y2]。
    :param angles1: 第一个脸的偏航角、俯仰角和滚转角。
    :param angles2: 第二个脸的偏航角、俯仰角和滚转角。
    :param distance_threshold: 脸部中心点之间的最大允许距离。
    :param size_difference_threshold: 最大允许大小差异。
    :param angle_difference_threshold: 最大允许角度差异。
    :return: 如果被认为是同一个脸，返回True，否则返回False。
    """
    # 检查中心点距离
    if not _are_boxes_close(box1, box2, distance_threshold):
        return False

    # 计算和比较面积
    area1 = _area_of_box(box1)
    area2 = _area_of_box(box2)
    if not _are_areas_similar(area1, area2, size_difference_threshold):
        return False

    # 检查角度相似性
    if not _are_angles_similar(angles1, angles2, angle_difference_threshold):
        return False

    return True

# 各模型针对不同度量的阈值设定
thresholds = {
    # "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86}, # 2622维
    "VGG-Face": {
        "cosine": 0.68,
        "euclidean": 1.17,
        "euclidean_l2": 1.17,
    },  # 4096维 - 使用LFW进行调整
    "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
    "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
    "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
    "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
    "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
    "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
    "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
    "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "euclidean_l2": 1.10},
}

def is_same_face_by_embedding(embedding1, embedding2) -> bool:
    # 计算向量的点积
    dot_product = np.dot(embedding1, embedding2)
    # 计算两个向量的模
    norm_vec1 = np.linalg.norm(embedding1)
    norm_vec2 = np.linalg.norm(embedding2)
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    # 将余弦相似度转换为余弦距离
    cosine_distance = 1 - cosine_similarity
    # print(f"\r\n\r\n\r\n\r\n face cosine_distance =======> {cosine_distance:.3f} \r\n\r\n\r\n\r\n")
    if cosine_distance < 0.6:
        return True
    return False
