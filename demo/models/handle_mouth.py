# 人脸眼睛检测处理
import numpy as np
import cv2  # 需要导入 cv2


class HandleMouth:
    def __init__(self):
        self.font_size = 0.5
        self.BGR = [0, 255, 0]

    def get_aspect_ratio(self, mouth):
        """
        计算嘴巴纵横比（MAR），用于判断嘴巴是否张开。
        :param mouth: 嘴巴区域的关键点列表（6个点）
        :return: 嘴巴纵横比值
        """
        # 计算嘴巴的垂直距离
        A = np.linalg.norm(mouth[2] - mouth[10])  # 上嘴唇与下嘴唇之间的垂直距离
        B = np.linalg.norm(mouth[4] - mouth[8])  # 上嘴唇与下嘴唇之间的另一垂直距离
        # 计算嘴巴的水平距离
        C = np.linalg.norm(mouth[0] - mouth[6])  # 嘴巴的宽度
        return (A + B) / (2.0 * C)

    def get_center(self, landmarks):
        """
        计算嘴巴区域的中心点
        :param landmarks: Dlib 68 个关键点
        :return: 嘴巴区域的中心点坐标
        """
        # 提取嘴巴的关键点（48到59点）
        mouth_points = []
        for i in range(48, 60):
            mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))

        # 计算嘴巴中心点
        mouth_center_x = int(np.mean([point[0] for point in mouth_points]))
        mouth_center_y = int(np.mean([point[1] for point in mouth_points]))
        return (mouth_center_x, mouth_center_y)

    def detect_mouth_open(self, frame, predictor, detector):
        """
        检测输入帧中的张嘴情况
        :param frame: 输入的图像帧
        :return: 嘴巴是否张开的标志（True：张嘴；False：闭嘴）
        """
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray)

        for face in faces:
            # 获取面部关键点
            landmarks = predictor(gray, face)
            # 计算嘴巴中心点
            mouth_center_point = self.get_center(landmarks)
            # 提取嘴巴的关键点
            mouth = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)]
            )

            # 计算 MAR（嘴巴纵横比）
            mar = self.get_aspect_ratio(mouth)

            # 设定 MAR 阈值，通常 0.5 以上为张嘴（需要根据实际情况调整）
            # 判断是否张嘴
            if mar > 0.5:
                cv2.putText(
                    frame,
                    "Mouth Open",
                    (mouth_center_point[0] - 150, mouth_center_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    (self.BGR),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Mouth Closed",
                    (mouth_center_point[0] - 150, mouth_center_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    (0, 0, 255),
                    2,
                )

            return frame  # 如果没有检测到人脸，默认为闭嘴
