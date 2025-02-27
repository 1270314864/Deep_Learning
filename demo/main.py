# main.py
import logging
from utils.camera import Camera  # 导入 Camera 类
from utils.logger import Logger
from detectors.detector_factory import DetectorFactory


def main():
    # 创建日志对象并获取 logger
    # log = Logger(log_file="app.log", log_level=logging.INFO)
    # logger = log.get_logger()
    # logger.info("初始化完成")
    # 创建 Camera 类的实例
    camera = Camera()
    # 获取摄像头对象
    video_capture = camera.create()
    if video_capture:
        hand_detector = DetectorFactory.get_detector("hand")
        face_detector = DetectorFactory.get_detector("face")
        # 使用 show_frame 函数处理每一帧
        camera.process_video(hand_detector.detect, face_detector.detect)


if __name__ == "__main__":
    main()
