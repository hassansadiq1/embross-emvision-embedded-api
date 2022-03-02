import time
from utilities.utils import FaceDetectionResult
from read_config_file import get_face_detection_sdk, get_camera_settings
from camera_module.camera import Camera
from utilities.utils import get_datetime
import copy

FaceDetection = None
selected_sdk = get_face_detection_sdk()
camera = get_camera_settings()

if selected_sdk == "roc":
    from face_detection_module.ROC.roc_model import ROCFaceDetection
    FaceDetection = ROCFaceDetection()
elif selected_sdk == "paravision":
    from face_detection_module.Paravision.paravision_model import ParavisionFaceDetection
    FaceDetection = ParavisionFaceDetection()
    # raise Exception("Paravision Model is currently not supported")
else:
    raise Exception("No such Face Detection SDK Found")


def detectFace():
    while True:
        FaceDetection.detect_face(Camera.get_current_frame())


def get_best_shot(duration):
    best_result = FaceDetectionResult()
    if duration == 0:
        if camera.liveness:
            return FaceDetection.get_liveness()
        else:
            return FaceDetection.get_faces(Camera.get_current_frame())
    else:
        start_time = time.time()
        time_elapsed = 0
        while time_elapsed < duration:
            result = FaceDetection.get_faces(Camera.get_current_frame())
            if best_result.quality < result.quality:
                best_result = copy.deepcopy(result)
            time_elapsed = time.time() - start_time
        best_result.time = get_datetime()
        return best_result
