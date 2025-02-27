# 人脸眼睛检测处理
import numpy as np
import cv2  # 需要导入 cv2


class HandleEyes:
    def __init__(self):
        self.font_size = 0.5
        self.BGR = [0, 255, 255]

    def get_aspect_ratio(self, eye):
        """
        计算眼睛纵横比（EAR），用于判断眼睛是否闭合。
        :param eye: 眼睛区域的关键点列表（6个点）
        :return: 眼睛纵横比值
        """
        A = np.linalg.norm(eye[1] - eye[5])  # 上眼睑与下眼睑的垂直距离
        B = np.linalg.norm(eye[2] - eye[4])  # 上眼睑与下眼睑的垂直距离
        C = np.linalg.norm(eye[0] - eye[3])  # 眼睛水平的宽度
        return (A + B) / (2.0 * C)

    def get_center(self, landmarks):
        """
        计算眼睛区域的中心点
        :param landmarks: Dlib 68 个关键点
        :return: 眼睛区域的中心点坐标
        """
        # 提取眼睛的关键点（36到41是左眼，42到47是右眼）
        left_eye = []
        right_eye = []
        for i in range(36, 42):
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        for i in range(42, 48):
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        # 计算左眼和右眼的中心点
        left_eye_center = (
            int(np.mean([point[0] for point in left_eye]) - 150),
            int(np.mean([point[1] for point in left_eye])),
        )
        right_eye_center = (
            int(np.mean([point[0] for point in right_eye]) - 150),
            int(np.mean([point[1] for point in right_eye])),
        )
        return left_eye_center, right_eye_center

    def detect_eyes_blink(self, frame, predictor, detector):
        """
        检测输入帧中的闭眼情况
        :param frame: 输入的图像帧
        :return: 眼睛是否闭合的标志（True：闭眼；False：睁眼）
        """
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = detector(gray)

        for face in faces:
            # 获取面部关键点
            landmarks = predictor(gray, face)
            left_eye_center, right_eye_center = self.get_center(landmarks)

            # 提取左眼和右眼的关键点
            left_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            )
            right_eye = np.array(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            )

            # 计算 EAR
            left_EAR = self.get_aspect_ratio(left_eye)
            right_EAR = self.get_aspect_ratio(right_eye)

            # 计算左右眼 EAR 的平均值
            ear = (left_EAR + right_EAR) / 2.0

            # 设定 EAR 阈值，通常 0.2~0.25 之间（需要根据实际情况调整）
            # 在眼睛中心处显示文本
            if ear < 0.15:
                cv2.putText(
                    frame,
                    "Eyes Close",
                    left_eye_center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Eyes Open",
                    left_eye_center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size,
                    (self.BGR),
                    2,
                )
        return frame
