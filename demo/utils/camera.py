import cv2
import logging
from utils.logger import Logger


class Camera:
    def __init__(self):
        # 检查CUDA是否可用
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("CUDA可用，启用GPU加速")
            # 使用CUDA加速的摄像头捕获
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
            # 设置CUDA加速
            self.video_capture.set(cv2.CAP_PROP_BACKEND, cv2.CAP_FFMPEG)
        else:
            print("CUDA不可用，使用CPU模式")
            self.video_capture = cv2.VideoCapture(0)

        # 设置摄像头的分辨率（宽度和高度）
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置宽度
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度

    def create(self):
        log = Logger(log_file="app.log", log_level=logging.INFO)
        logger = log.get_logger()
        logger.info("摄像头已创建")
        # 检查摄像头是否成功打开
        if not self.video_capture.isOpened():
            print("无法打开摄像头")
            return None
        return self.video_capture

    def get_frame(self):
        # 获取视频帧
        ret, frame = self.video_capture.read()
        if not ret:
            print("无法读取视频帧")
            return None

        # 如果使用GPU，将图像转换为GPU内存
        if self.use_gpu:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            # 在GPU上进行图像处理
            frame = gpu_frame.download()

        return frame

    def release(self):
        # 释放摄像头资源
        self.video_capture.release()

    def process_video(self, *detects):
        """处理视频流并执行传入的操作。
        :param action: 一个函数，接受当前帧并执行操作。
        """
        while True:
            frame = self.get_frame()  # 获取当前帧
            if frame is None:
                break

            # 依次执行传入的所有操作
            for detect in detects:
                frame = detect(frame)

            cv2.imshow("video", frame)
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 释放摄像头资源
        self.release()
        cv2.destroyAllWindows()
