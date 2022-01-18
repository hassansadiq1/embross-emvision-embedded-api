import threading
from camera_module.camera import Camera
import cv2

thread_lock = threading.Lock()


def generate_camera_stream():
    # grab global references to the output frame and lock variables
    # loop over frames from the output stream
    while True:
        frame = Camera.get_current_frame()
        if frame is None:
            continue
        # locks thread until encoding finishes
        with thread_lock:
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
