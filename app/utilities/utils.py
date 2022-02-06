from pydantic import BaseModel
from PIL import Image as Pil
from typing import Optional
from math import sqrt
import base64
import numpy as np
import cv2
import io
from datetime import datetime

software_version = "ENA-1.2.4"


class CameraSettings(BaseModel):
    id: int = 0
    name: str = None
    frames_per_sec: float = 30.0
    frame_width: int = 1280
    frame_height: int = 720
    brightness: int = 0
    contrast: int = 36
    hue: int = 0
    saturation: int = 72
    sharpness: int = 2
    gamma: int = 100
    backlight: int = 1
    rotation: int = 0
    online: bool = False
    enabled: bool = True
    liveness: int = 0


class AppConfig:
    def __init__(self, camera_config: CameraSettings, host_url: str, port_num: int, face_detection_model: str):
        self.camera_config = camera_config
        self.host_url = host_url
        self.port_num = port_num
        self.face_detection_model = face_detection_model

    def get_camera_configuration(self):
        return self.camera_config

    def get_host_url(self):
        return self.host_url

    def get_port_num(self):
        return self.port_num

    def get_face_detection_model(self):
        return self.face_detection_model


# Detected face image Format
class FaceDetectionResult(BaseModel):
    quality: float = 0
    acceptability: float = 0
    face_size: float = 0
    liveness: float = 0
    base64 = "null"
    faces: int = 0
    box_x: int = 0
    box_y: int = 0
    box_height: int = 0
    box_width: int = 0
    time: datetime = None


class ImageComparision(BaseModel):
    score: float = 0
    image1_face_quality: float = 0
    image1_number_of_faces: int = 0
    image2_face_quality: float = 0
    image2_number_of_faces: int = 0
    time: datetime = 0


class CroppedFaceSettings(BaseModel):
    face_size_threshold: int = 0
    height_padding: int = 0
    width_padding: int = 0
    liveness_window: int = 5


# Raw camera image format
class RawFrame(BaseModel):
    base64 = "null"
    time: datetime = None


class VideoStream(BaseModel):
    url = "null"
    time: datetime = None


class Camera(BaseModel):
    name: str = None
    online: bool
    enabled: bool
    resolution: str = None


# software information format
# will be removed if not needed, more info will be added if needed
class SoftwareInfo(BaseModel):
    version = "null"


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserInDB(User):
    hashed_password: str


class LogConfig:
    level: str
    backup_in_days: int


# converts image to base64 string format
def convert_image_to_base64(_image):
    return_value, out_frame = cv2.imencode('.jpg', _image)
    return base64.b64encode(out_frame)


# Take in base64 string and return CV image
def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    pil_image = Pil.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)


# crops face from image
# returns cropped image
def crop_face(top_left_x, top_left_y, bottom_right_x, bottom_right_y,  _input_frame):
    # crops image           [start_row:end_row, start_column:end_column]
    crop_img = _input_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return crop_img


def get_datetime():
    return datetime.now()


def get_software_info():
    return software_version


def get_distance_between_two_pints(x1, y1, x2, y2):
    return sqrt(pow(x2 - x1, 2) +
                pow(y2 - y1, 2) * 1.0)
