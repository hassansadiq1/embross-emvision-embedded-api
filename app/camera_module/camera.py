import cv2
import threading
from utilities.utils import RawFrame, get_datetime, convert_image_to_base64, CameraSettings
from face_detection_module.Paravision.paravision_helper import FaceProcessor
import numpy as np
import pyrealsense2 as rs
import time
from actuator.actuator import Actuator

outputFrame = None
depthFrame = None
perform_detection = False
thread_lock = threading.Lock()
camera_status = None
camera_params = None
color = (255, 0, 0)
thickness = 2


class Camera:
    @staticmethod
    def get_current_frame():
        return outputFrame

    @staticmethod
    def set_perform_detection():
        global perform_detection
        perform_detection = True

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
        self.FaceDetection = None
        # self.FaceDetection = FaceProcessor()
        self.actuator = Actuator()

    def initialize(self):
        # self.cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
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

    def runCam(self):
        global camera_status
        global perform_detection
        while True:
            if self.stop:
                self.cap.release()
                return

            if self.cap.isOpened():
                read_status, frame = self.cap.read()
                if read_status:
                    camera_status = True
                    if self.camera_config.rotation == 90:
                        frame = cv2.rotate(frame, 0)
                    elif self.camera_config.rotation == 180:
                        frame = cv2.rotate(frame, 1)
                    elif self.camera_config.rotation == 270:
                        frame = cv2.rotate(frame, 2)

                    if perform_detection:
                        face_result = self.FaceDetection.detect_faces(frame)
                        print(face_result)
                        # use face result to move actuator

                        if self.FaceDetection.verify_face_position():
                            print("stop actuator and perform next steps")
                            self.FaceDetection.num_faces = 0
                            perform_detection = False

                    with thread_lock:
                        global outputFrame
                        outputFrame = frame.copy()
                        if self.FaceDetection.num_faces:
                            outputFrame = cv2.rectangle(outputFrame,
                                                        (self.FaceDetection.top_left_x, self.FaceDetection.top_left_y),
                                                        (self.FaceDetection.bottom_right_x,
                                                         self.FaceDetection.bottom_right_y),
                                                        color, thickness)
                else:
                    self.cap.release()
                    camera_status = False
            else:
                camera_status = False
                time.sleep(0.2)
                self.initialize()

    def initializeRealsense(self):

        global camera_status
        camera_status = False
        ctx = rs.context()
        if len(ctx.devices) > 0:
            # resetting device
            # devices = ctx.query_devices()
            # for dev in devices:
            #     dev.hardware_reset()

            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break

            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                camera_status = False
                return False

            config.enable_stream(rs.stream.depth,
                                 self.camera_config.frame_width, self.camera_config.frame_height,
                                 rs.format.z16, self.camera_config.frames_per_sec)
            config.enable_stream(rs.stream.color,
                                 self.camera_config.frame_width, self.camera_config.frame_height,
                                 rs.format.bgr8, self.camera_config.frames_per_sec)

            # Start streaming
            profile = None
            profile = self.pipeline.start(config)
            if profile:
                print("camera is live")
            else:
                camera_status = False
                print("unable to initialize camera")
                return False

            # setting camera properties
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.brightness, self.camera_config.brightness)
            color_sensor.set_option(rs.option.contrast, self.camera_config.contrast)
            color_sensor.set_option(rs.option.hue, self.camera_config.hue)
            color_sensor.set_option(rs.option.saturation, self.camera_config.saturation)
            color_sensor.set_option(rs.option.sharpness, self.camera_config.sharpness)
            color_sensor.set_option(rs.option.gamma, self.camera_config.gamma)
            color_sensor.set_option(rs.option.backlight_compensation, self.camera_config.backlight)

            depth_stream = profile.get_stream(rs.stream.depth)
            color_stream = profile.get_stream(rs.stream.color)
            depth_profile = depth_stream.as_video_stream_profile()
            color_profile = color_stream.as_video_stream_profile()
            depth_intr = depth_profile.get_intrinsics()
            color_intr = color_profile.get_intrinsics()
            color_to_depth_extr = color_profile.get_extrinsics_to(depth_stream)
            global camera_params
            from paravision.liveness import CameraParams
            camera_params = CameraParams(depth_intr, color_intr, color_to_depth_extr)
            camera_status = True
        return camera_status

    def runRealsenseCam(self):
        global camera_status
        global outputFrame, depthFrame
        global perform_detection

        while True:
            self.actuator.exercise_actuator()
            if self.stop:
                self.pipeline.stop()
                return

            try:
                if camera_status:
                    frames = self.pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    if not depth_frame or not color_frame:
                        camera_status = False
                        self.initialize()
                        continue
                    if self.camera_config.rotation == 90:
                        color_image = cv2.rotate(color_image, 0)
                        depth_image = cv2.rotate(depth_image, 0)
                    elif self.camera_config.rotation == 180:
                        color_image = cv2.rotate(color_image, 1)
                        depth_image = cv2.rotate(depth_image, 1)
                    elif self.camera_config.rotation == 270:
                        color_image = cv2.rotate(color_image, 2)
                        depth_image = cv2.rotate(depth_image, 2)

                    if perform_detection:
                        face_result = self.FaceDetection.detect_faces(color_image)
                        # self.actuator.move_actuator_to(face_result)
                        print(face_result)
                        # use face result to move actuator

                        if self.FaceDetection.verify_face_position():
                            print("accumulating frames for liveness")
                            # stop actuator and perform liveness
                            self.FaceDetection.get_liveness(camera_params, depth_image)
                            if self.FaceDetection.livenessFlag:
                                self.FaceDetection.window.clear()
                                self.FaceDetection.livenessFlag = False
                                if self.FaceDetection.face_result.liveness > 0:
                                    print("liveness passed, take 4k picture here")
                                else:
                                    print("liveness test failed")
                                perform_detection = False
                                self.FaceDetection.face_position_counter = 0
                                self.FaceDetection.num_faces = 0
                        else:
                            # move actuator until face position is fixed
                            pass

                    with thread_lock:
                        if self.FaceDetection.num_faces:
                            outputFrame = cv2.rectangle(outputFrame,
                                                        (self.FaceDetection.top_left_x, self.FaceDetection.top_left_y),
                                                        (self.FaceDetection.bottom_right_x,
                                                         self.FaceDetection.bottom_right_y),
                                                        color, thickness)
                        outputFrame = color_image.copy()

                else:
                    time.sleep(0.2)
                    self.initializeRealsense()
            except:
                self.pipeline.stop()
                camera_status = False

    def run(self):
        self.FaceDetection = FaceProcessor()
        if self.camera_config.liveness:
            self.initializeRealsense()
            self.runRealsenseCam()
        else:
            self.initialize()
            self.runCam()
        return
