import cv2
import threading
from utilities.utils import RawFrame, get_datetime, convert_image_to_base64, CameraSettings
from time import sleep

outputFrame = None
thread_lock = threading.Lock()
camera_status = None


class Camera:
    @staticmethod
    def get_current_frame():
        return outputFrame

    @staticmethod
    def get_base64_image():
        raw_frame = RawFrame()
        raw_frame.time = get_datetime()
        raw_frame.base64 = convert_image_to_base64(outputFrame)
        return raw_frame

    @staticmethod
    def get_camera_status():
        return camera_status


# This thread captures image continuously
class CameraThread(threading.Thread):
    def __init__(self, camera_config: CameraSettings):
        threading.Thread.__init__(self)
        self.camera_config = camera_config
        self.cap = None
        self.stop = False

    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera_config.id)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.frames_per_sec)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.frame_height)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_config.brightness)
        self.cap.set(cv2.CAP_PROP_CONTRAST, self.camera_config.contrast)
        self.cap.set(cv2.CAP_PROP_HUE, self.camera_config.hue)
        self.cap.set(cv2.CAP_PROP_SATURATION, self.camera_config.saturation)
        self.cap.set(cv2.CAP_PROP_SHARPNESS, self.camera_config.sharpness)
        self.cap.set(cv2.CAP_PROP_GAMMA, self.camera_config.gamma)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, self.camera_config.backlight)

    def run(self):
        global camera_status
        self.initialize()
        while True:
            if self.stop:
                self.cap.release()
                return

            if self.cap.isOpened():
                read_status, frame = self.cap.read()
                if read_status:
                    camera_status = True
                    with thread_lock:
                        if self.camera_config.rotation == 90:
                            frame = cv2.rotate(frame, 0)
                        elif self.camera_config.rotation == 180:
                            frame = cv2.rotate(frame, 1)
                        elif self.camera_config.rotation == 270:
                            frame = cv2.rotate(frame, 2)
                        global outputFrame
                        outputFrame = frame.copy()
                else:
                    self.cap.release()
                    camera_status = False
            else:
                camera_status = False
                self.initialize()
