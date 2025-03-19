import logging
from utils.logger import Logger
import cv2
import face_recognition
import dlib
from imutils import face_utils
import numpy as np

from models.handle_mouth import HandleMouth
from models.handle_eyes import HandleEyes


class FaceDetector:
    def __init__(self):
        self.log = Logger(log_file="app.log", log_level=logging.INFO)
        self.logger = self.log.get_logger()
        self.logger.info(f"创建 人脸创建器")

        # 检查CUDA是否可用
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            self.logger.info("CUDA可用，启用GPU加速")
            # 使用CUDA加速的dlib检测器
            self.detector = dlib.cuda.get_frontal_face_detector()
            self.predictor = dlib.cuda.shape_predictor(
                "./resource/data/shape_predictor_68_face_landmarks.dat"
            )
        else:
            self.logger.info("CUDA不可用，使用CPU模式")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(
                "./resource/data/shape_predictor_68_face_landmarks.dat"
            )

        self.face_locations = []  # 存储人脸的位置
        self.face_encodings = []  # 存储人脸编码
        self.face_names = []  # 存储匹配到的人脸名称
        self.process_this_frame = True  # 控制每帧是否处理（优化性能）
        self.known_face_encodings = [
            load_face_encoding("./resource/image/obama.jpg"),
            load_face_encoding("./resource/image/biden.jpg"),
            load_face_encoding("./resource/image/jlw_face.jpg"),
        ]
        self.known_face_names = ["Barack Obama", "Joe Biden", "Liang wei Jiang"]

    def detect(self, frame):
        if frame is None:
            print("Failed to capture image.")
            exit(1)

        # 如果使用GPU，将图像转换为GPU内存
        if self.use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            gray = gray.download()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 处理张嘴闭嘴
        handleMouth = HandleMouth()
        handleMouth.detect_mouth_open(frame, self.predictor, self.detector)
        # 处理人眨眼
        handleEyes = HandleEyes()
        handleEyes.detect_eyes_blink(frame, self.predictor, self.detector)

        # 仅处理每隔一帧的视频，以节省时间
        if self.process_this_frame:
            # 将视频帧缩小为 1/4 大小，以加快人脸识别处理速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # 将图像从 BGR 颜色空间转换为 RGB 颜色空间
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # 使用GPU加速的人脸检测
            if self.use_gpu:
                # 将图像上传到GPU
                gpu_small_frame = cv2.cuda_GpuMat()
                gpu_small_frame.upload(rgb_small_frame)
                # 在GPU上进行人脸检测
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations
                )
            else:
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations
                )

            self.face_names = []
            for face_encoding in self.face_encodings:
                # 检查该面部是否与已知面部匹配
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding
                )
                name = "Unknown"  # 默认名称是未知
                # 或者，使用与新面部最小距离的已知面部
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame  # 控制处理每帧的开关

        # 显示结果
        for (top, right, bottom, left), name in zip(
            self.face_locations, self.face_names
        ):
            # 由于我们检测到的图像已经缩小为 1/4 大小，需要将人脸位置放大回原来的大小
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 在人脸下方绘制带有名称的标签
            cv2.rectangle(
                frame,
                (left, bottom + 10),
                (right + 50, bottom + 50),
                (0, 0, 255),
                cv2.FILLED,
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom + 40), font, 1.0, (255, 255, 255), 1
            )

        rects = self.detector(gray, 0)  # 检测图像中的人脸
        for i, rect in enumerate(rects):
            shape = self.predictor(gray, rect)  # 获取面部关键点
            shape = face_utils.shape_to_np(shape)  # 将面部关键点转换为 NumPy 数组
            self.draw_landmarks(frame, shape)  # 调用绘制关键点的方法

        return frame

    def draw_landmarks(self, frame, shape):
        # 循环遍历面部关键点并在图像上绘制
        for idx, (x, y) in enumerate(shape):
            landmark = idx
            if 48 <= landmark < 60:  # 根据编号选择嘴巴区域绘制
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            if 36 <= landmark < 48:  # 根据编号选择眼睛区域绘制
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)


def load_face_image_file(path):
    face_image = face_recognition.load_image_file(path)
    if face_image is None:
        print(f"{(path)} Failed to load image.")
    else:
        print(f"{(path)} Image loaded successfully.")
    return face_image


def load_face_encoding(path):
    image = load_face_image_file(path)
    face_encoding = face_recognition.face_encodings(image)[0]
    return face_encoding
