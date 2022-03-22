from read_config_file import get_face_settings, get_camera_settings
import numpy as np
import utilities.utils as utils


class FaceProcessor:
    def __init__(self):
        from paravision.recognition import Session, Engine
        from paravision.liveness.session import Liveness
        self.face_config = get_face_settings()
        self.session = Session(engine=Engine.TENSORRT)
        self.camera_config = get_camera_settings()
        self.session.get_faces([np.random.rand(self.camera_config.frame_width, self.camera_config.frame_height, 3)],
                               qualities=True)
        self.liveness = None
        if self.camera_config.liveness:
            self.liveness = Liveness(settings={"max_batch_size": 1})

        # detection results
        self.detection_result = None
        self.top_left_x = 0
        self.top_left_y = 0
        self.bottom_right_x = 0
        self.bottom_right_y = 0

    def detect_faces(self, _in_image):
        self.detection_result = self.session.get_faces([_in_image], qualities=True)
        if len(self.detection_result.faces) > 0:
            self.top_left_x = int(self.detection_result.faces[0].bounding_box.top_left.x)
            self.top_left_y = int(self.detection_result.faces[0].bounding_box.top_left.y)
            self.bottom_right_x = int(self.detection_result.faces[0].bounding_box.bottom_right.x)
            self.bottom_right_y = int(self.detection_result.faces[0].bounding_box.bottom_right.y)
            return True
        return False
