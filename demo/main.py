import logging
from utils.camera import Camera
from utils.logger import Logger
from detectors.detector_factory import DetectorFactory
import cv2


def init_detectors():
    """初始化检测器"""
    return (DetectorFactory.get_detector("hand"), DetectorFactory.get_detector("face"))


def main():
    # 初始化日志
    log = Logger(log_file="app.log", log_level=logging.INFO)
    logger = log.get_logger()
    logger.info("程序启动")

    try:
        # 初始化摄像头
        camera = Camera()
        camera.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 降低到640x480
        camera.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture = camera.create()
        if not video_capture:
            logger.error("摄像头初始化失败")
            return

        # 初始化检测器
        hand_detector, face_detector = init_detectors()

        # 处理视频流
        logger.info("开始处理视频流")
        camera.process_video(hand_detector.detect, face_detector.detect)
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
