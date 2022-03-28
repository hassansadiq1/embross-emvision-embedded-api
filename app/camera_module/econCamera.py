import cv2
from app.utilities.utils import RawFrame, convert_image_to_base64, get_datetime, CameraSettings


output_image = None


def get_econ_image():
    return output_image


def clear_frame():
    global output_image
    output_image = None


def get_econ_base64():
    raw_frame = RawFrame()
    raw_frame.time = get_datetime()
    if output_image is not None:
        raw_frame.base64 = convert_image_to_base64(output_image)
    return raw_frame


class EconCamera:
    def __init__(self, _config: CameraSettings):
        self._config = _config
        self.cap = cv2.VideoCapture(self._config.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, self._config.frames_per_sec)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

    def capture_image(self):
        res = False
        if self.cap.isOpened():
            read_status, frame = self.cap.read()
            if read_status:
                if self._config.rotation == 90:
                    frame = cv2.rotate(frame, 0)
                elif self._config.rotation == 180:
                    frame = cv2.rotate(frame, 1)
                elif self._config.rotation == 270:
                    frame = cv2.rotate(frame, 2)
                res = True
                global output_image
                output_image = frame.copy()
            else:
                output_image = None
                self.cap.release()
        else:
            res = False
            output_image = None
            self.cap = cv2.VideoCapture(self._config.id, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FPS, self._config.frames_per_sec)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        return res














