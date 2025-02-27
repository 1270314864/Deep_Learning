import logging
from utils.logger import Logger

from queue import Full
from .face_detector import FaceDetector
from .hand_detector import HandDetector


class DetectorFactory:
    @staticmethod
    def get_detector(detector_type):
        if detector_type == "face":
            return FaceDetector()
        elif detector_type == "hand":
            return HandDetector()
