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

        self.face_position_counter = 0

        # detection results
        self.detection_result = None
        self.face_result = utils.FaceDetectionResult()
        self.top_left_x = 0
        self.top_left_y = 0
        self.bottom_right_x = 0
        self.bottom_right_y = 0

    def detect_faces(self, _in_image):
        self.face_result = utils.FaceDetectionResult()
        self.detection_result = self.session.get_faces([_in_image], qualities=True)
        if len(self.detection_result.faces) > 0:
            # Extract left and right eye positions
            left_eye_x = self.detection_result.faces[0].landmarks.left_eye.x
            left_eye_y = self.detection_result.faces[0].landmarks.left_eye.y
            right_eye_x = self.detection_result.faces[0].landmarks.right_eye.x
            right_eye_y = self.detection_result.faces[0].landmarks.right_eye.y
            # find distance between eyes. This will help to determine the distance from user to kiosk
            face_size = utils.get_distance_between_two_pints(left_eye_x, left_eye_y, right_eye_x, right_eye_y)

            self.top_left_x = int(self.detection_result.faces[0].bounding_box.top_left.x)
            self.top_left_y = int(self.detection_result.faces[0].bounding_box.top_left.y)
            self.bottom_right_x = int(self.detection_result.faces[0].bounding_box.bottom_right.x)
            self.bottom_right_y = int(self.detection_result.faces[0].bounding_box.bottom_right.y)

            self.face_result.faces = len(self.detection_result.faces)
            self.face_result.quality = self.detection_result.faces[0].quality
            self.face_result.acceptability = self.detection_result.faces[0].acceptability
            self.face_result.liveness = -1
            self.face_result.face_size = face_size
            self.face_result.box_height = self.detection_result.faces[0].bounding_box.height()
            self.face_result.box_width = self.detection_result.faces[0].bounding_box.width()
            self.face_result.box_x = int(self.detection_result.faces[0].bounding_box.top_left.x)
            self.face_result.box_y = int(self.detection_result.faces[0].bounding_box.top_left.y)
            self.face_result.top_left_x = int(self.detection_result.faces[0].bounding_box.top_left.x)
            self.face_result.top_left_y = int(self.detection_result.faces[0].bounding_box.top_left.y)
            self.face_result.bottom_right_x = int(self.detection_result.faces[0].bounding_box.bottom_right.x)
            self.face_result.bottom_right_y = int(self.detection_result.faces[0].bounding_box.bottom_right.y)
            self.face_result.frame_height = _in_image.shape[0]
            self.face_result.frame_width = _in_image.shape[1]
        return self.face_result

    def check_face_size(self):
        left_eye_x = self.detection_result.faces[0].landmarks.left_eye.x
        left_eye_y = self.detection_result.faces[0].landmarks.left_eye.y
        right_eye_x = self.detection_result.faces[0].landmarks.right_eye.x
        right_eye_y = self.detection_result.faces[0].landmarks.right_eye.y
        # find distance between eyes. This will help to determine the distance from user to kiosk
        face_size = utils.get_distance_between_two_pints(left_eye_x, left_eye_y, right_eye_x, right_eye_y)
        if face_size > self.face_config.face_size_threshold:
            return True
        else:
            return False

    def verify_face_position(self):
        self.face_position_counter += 1
        if self.face_position_counter > 30:
            self.face_position_counter = 0
            return True
        else:
            return False
