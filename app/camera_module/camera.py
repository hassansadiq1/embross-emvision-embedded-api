import cv2
import threading
from utilities.utils import RawFrame, get_datetime, convert_image_to_base64, CameraSettings
import numpy as np
import time

outputFrame = None
depthFrame = None
thread_lock = threading.Lock()
camera_status = None
camera_params = None


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

    @staticmethod
    def get_current_depth_frame():
        return outputFrame, depthFrame

    @staticmethod
    def get_camera_params():
        return camera_params


# This thread captures image continuously
class CameraThread(threading.Thread):
    def __init__(self, camera_config: CameraSettings):
        threading.Thread.__init__(self)
        self.camera_config = camera_config
        self.cap = None
        self.stop = False
        self.pipeline = None

        self.initialize_econ()

    def gstreamer_pipeline(self,
            device="/dev/video3", frame_rate=15,
            capture_width=3840, capture_height=2160,
            brightness=0, contrast=36,
            hue=0, saturation=72
    ):
        return (
                "v4l2src device=%s num-buffers=450 "
                "brightness=%d contrast=%d hue=%d saturation=%d ! "
                "video/x-raw, "
                "width=(int)%d, height=(int)%d, "
                "framerate=(fraction)%d/1 ! "
                "videoconvert ! appsink"
                % (device,
                   brightness, contrast,
                   hue, saturation,
                   capture_width,
                   capture_height, frame_rate)
        )

    def initialize_econ(self):
        print("************* Initializing Camera ******************")
        self._frame = None
        pipeline = self.gstreamer_pipeline(device=self.camera_config.id,
                                           frame_rate=self.camera_config.frames_per_sec,
                                           capture_width=self.camera_config.frame_width,
                                           capture_height=self.camera_config.frame_height,
                                           brightness=self.camera_config.brightness,
                                           contrast=self.camera_config.contrast,
                                           hue=self.camera_config.hue,
                                           saturation=self.camera_config.saturation)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # self.cap.set(cv2.CAP_PROP_FPS, self._config.frames_per_sec)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

    def runEconCamera(self):
        global camera_status

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
                time.sleep(0.2)
                self.initialize_econ()


    def run(self):
        if self.camera_config.name == "econ":
            # self.initialize_econ()
            self.runEconCamera()
        return
