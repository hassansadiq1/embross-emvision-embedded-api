import time
from app.utilities.utils import FaceDetectionResult
from app.read_config_file import get_face_detection_sdk
from app.camera_module.camera import Camera
from app.utilities.utils import get_datetime
import copy


FaceDetection = None
selected_sdk = get_face_detection_sdk()

if selected_sdk == "roc":
    from app.face_detection_module.ROC.roc_model import ROCFaceDetection
    FaceDetection = ROCFaceDetection()
elif selected_sdk == "paravision":
    from app.face_detection_module.Paravision.paravision_model import ParavisionFaceDetection
    FaceDetection = ParavisionFaceDetection()
    # raise Exception("Paravision Model is currently not supported")
else:
    raise Exception("No such Face Detection SDK Found")



def get_best_shot(duration):
    best_result = FaceDetectionResult()
    if duration == 0:
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
