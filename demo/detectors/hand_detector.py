import cv2
import mediapipe as mp  # type: ignore
from models.handle_gesture import HandleGesture


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame):
        if frame is None:
            print("Failed to capture image.")
        with self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as hands:
            # 转换为 RGB 格式，因为 MediaPipe 需要 RGB 输入
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 处理帧图像w
            results = hands.process(image)

            # 将图像转换回 BGR 格式以便 OpenCV 显示
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 如果检测到手部
            if results.multi_hand_landmarks:

                handleGesture = HandleGesture()
                for landmarks in results.multi_hand_landmarks:
                    handleGesture.is_fist_gesture(frame, landmarks, self.mp_hands)
                    handleGesture.is_ok_gesture(frame, landmarks, self.mp_hands)

                    # 绘制手部的关键点
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    # 可以访问手部关键点坐标，关键点是从 0 到 20 的 21 个点
                    for id, landmark in enumerate(landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(
                            frame,
                            str(id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
        return frame
