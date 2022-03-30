import cv2
from utilities.utils import RawFrame, convert_image_to_base64, get_datetime, CameraSettings


output_image = None


def get_econ_image():
    return output_image


def clear_frame():
    global output_image
    if output_image is not None:
        output_image = None


def get_econ_base64():
    raw_frame = RawFrame()
    raw_frame.time = get_datetime()
    if output_image is not None:
        raw_frame.base64 = convert_image_to_base64(output_image)
    return raw_frame


def gstreamer_pipeline(
        device="/dev/video3", frame_rate=15,
        capture_width=3840, capture_height=2160):
    return (
            "v4l2src device=%s num-buffers=450 ! "
            "video/x-raw, "
            "width=(int)%d, height=(int)%d, "
            "framerate=(fraction)%d/1 ! "
            "videoconvert ! appsink"
            % (device, capture_width,
               capture_height, frame_rate)
    )


class EconCamera:
    def __init__(self, _config: CameraSettings):
        self._config = _config
        pipeline = gstreamer_pipeline(device=self._config.id,
                                      frame_rate=self._config.frames_per_sec,
                                      capture_width=self._config.frame_width,
                                      capture_height=self._config.frame_height)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        # self.cap.set(cv2.CAP_PROP_FPS, self._config.frames_per_sec)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

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
                cv2.imwrite("test.jpg", output_image)
            else:
                output_image = None
                self.cap.release()
        else:
            res = False
            output_image = None
            pipeline = gstreamer_pipeline(device=self._config.id,
                                          frame_rate=self._config.frames_per_sec,
                                          capture_width=self._config.frame_width,
                                          capture_height=self._config.frame_height)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            # self.cap.set(cv2.CAP_PROP_FPS, self._config.frames_per_sec)
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        return res
