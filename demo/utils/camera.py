import cv2
import logging
from utils.logger import Logger


class Camera:
    def __init__(self):

        # 初始化摄像头，默认使用摄像头 0
        self.video_capture = cv2.VideoCapture(0)
        # 设置摄像头的分辨率（宽度和高度）
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 设置宽度
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 设置高度

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
