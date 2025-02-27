import cv2  # 需要导入 cv2
import math


class HandleGesture:
    def __init__(self):
        self.font_size = 0.5
        self.BGR = [0, 255, 0]

    def get_center(self, frame, landmarks, mp_hands):
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # 计算手部中心位置（可以是手指指尖的平均位置）
        center_x = (
            thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x
        ) / 5
        center_y = (
            thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y
        ) / 5
        # 将手部中心点作为文本位置
        return (int(center_x * frame.shape[1]), int(center_y * frame.shape[0]))

    def points_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def is_ok_gesture(self, frame, landmarks, mp_hands):
        """
        判断是否为单手OK手势
        """
        try:
            # 获取拇指和食指指尖的坐标
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # 计算拇指和食指指尖的距离
            distance = (
                (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
            ) ** 0.5

            # 判断是否为OK手势
            if distance < 0.05:  # 根据实际情况调整阈值
                text = "OK Gesture Detected"
                color = (0, 255, 0)  # 绿色
            else:
                text = "No OK Gesture Detected"
                color = (0, 0, 255)  # 红色

            text_position = self.get_center(frame, landmarks, mp_hands)

            # 在视频帧上绘制文本
            cv2.putText(
                frame,
                text,
                (text_position[0], text_position[1] - 10),  # 在手腕上方显示文本
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        except AttributeError as e:
            print(f"Error accessing thumb landmarks: {e}")
            return frame

    def is_heart_gesture(self, frame, hand_landmarks, mp_hands):
        """
        判断是否为单手比心手势
        """
        try:
            # 获取手指的关节信息
            landmarks = hand_landmarks.landmark
            # 默认认为是比心手势
            heart = True

            # 拇指关节检测（关节：0, 1, 2）
            # 拇指的MCP、IP和TIP关节的y坐标需要满足一定条件来形成比心手势
            if (
                landmarks[mp_hands.HandLandmark.THUMB_CMC].y
                < landmarks[mp_hands.HandLandmark.THUMB_IP].y
            ):
                heart = False
            if (
                landmarks[mp_hands.HandLandmark.THUMB_IP].y
                < landmarks[mp_hands.HandLandmark.THUMB_TIP].y
            ):
                heart = False

            # 食指关节检测（关节：5, 6, 7）
            # 食指的MCP、PIP、DIP和TIP关节的y坐标需要满足一定条件来形成比心手势
            if (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            ):
                heart = False
            if (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
            ):
                heart = False
            if (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            ):
                heart = False

            # 判断拇指和食指的相对位置，确保拇指和食指形成"心形"
            if (
                landmarks[mp_hands.HandLandmark.THUMB_TIP].x
                > landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            ):
                heart = False  # 拇指应该位于食指的左侧（假设手掌是面向摄像头的）

            # 其他手指不需要参与比心手势，可以保持自然伸展
            # 这里我们假设其他手指（中指、无名指、小指）自然伸展即可，不做严格的检测。

            if heart:
                text = "Heart Gesture Detected"
                color = (0, 0, 255)
            else:
                text = "No Heart Gesture Detected"
                color = (0, 0, 255)

            text_position = self.get_center(frame, hand_landmarks, mp_hands)

            # 在视频帧上绘制文本
            cv2.putText(
                frame,
                text,
                (text_position[0], text_position[1] - 10),  # 在手腕上方显示文本
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        except AttributeError as e:
            print(f"Error accessing thumb landmarks: {e}")
            return frame

    def is_fist_gesture(self, frame, hand_landmarks, mp_hands):
        """
        判断是否为握拳手势
        """
        try:
            # 获取手指的关节信息
            landmarks = hand_landmarks.landmark
            # 默认认为是握拳
            fist = True

            # 拇指关节检测（关节：0, 1, 2）
            if (
                landmarks[mp_hands.HandLandmark.THUMB_CMC].y
                < landmarks[mp_hands.HandLandmark.THUMB_IP].y
                or landmarks[mp_hands.HandLandmark.THUMB_IP].y
                < landmarks[mp_hands.HandLandmark.THUMB_TIP].y
            ):
                fist = True
            else:
                fist = False

            # 食指关节检测（关节：5, 6, 7）
            if (
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                or landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                or landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            ):
                fist = True
            else:
                fist = False

            # 中指关节检测（关节：9, 10, 11）
            if (
                landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                or landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                or landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            ):
                fist = True
            else:
                fist = False

            # 无名指关节检测（关节：13, 14, 15）
            if (
                landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y
                < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
                or landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
                < landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
                or landmarks[mp_hands.HandLandmark.RING_FINGER_DIP].y
                < landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
            ):
                fist = True
            else:
                fist = False

            # 小指关节检测（关节：17, 18, 19）
            if (
                landmarks[mp_hands.HandLandmark.PINKY_MCP].y
                < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
                or landmarks[mp_hands.HandLandmark.PINKY_PIP].y
                < landmarks[mp_hands.HandLandmark.PINKY_DIP].y
                or landmarks[mp_hands.HandLandmark.PINKY_DIP].y
                < landmarks[mp_hands.HandLandmark.PINKY_TIP].y
            ):
                fist = True
            else:
                fist = False

            text_position = self.get_center(frame, hand_landmarks, mp_hands)

            if fist:
                text = "Fist Gesture Detected"
                color = (0, 255, 0)
            else:
                text = "No Fist Gesture Detected"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (text_position[0], text_position[1] - 10),  # 在手腕上方显示文本
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        except AttributeError as e:
            print(f"Error accessing thumb landmarks: {e}")
            return frame
