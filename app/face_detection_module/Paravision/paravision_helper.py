from read_config_file import get_face_settings, get_camera_settings
import numpy as np
import utilities.utils as utils


class FaceProcessor:
    def __init__(self):
        from paravision.recognition import Session, Engine
        from paravision.liveness.session import Liveness
        self.FACE_CONFIG = get_face_settings()
        self.session = Session(engine=Engine.TENSORRT)
        self.camera_config = get_camera_settings()
        self.session.get_faces([np.random.rand(self.camera_config.frame_width, self.camera_config.frame_height, 3)],
                               qualities=True)
        self.liveness = None
        if self.camera_config.liveness:
            self.liveness = Liveness(settings={"max_batch_size": 1})

        self.face_position_counter = 0

        # detection results
        self.num_faces = 0
        self.detection_result = None
        self.face_result = utils.FaceDetectionResult()
        self.top_left_x = 0
        self.top_left_y = 0
        self.bottom_right_x = 0
        self.bottom_right_y = 0

        # liveness
        self.window = []
        self.livenessFlag = False

    def detect_faces(self, _in_image):
        self.face_result = utils.FaceDetectionResult()
        self.detection_result = self.session.get_faces([_in_image], qualities=True)
        self.num_faces = len(self.detection_result.faces)
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
        if face_size > self.FACE_CONFIG.face_size_threshold:
            return True
        else:
            return False

    def verify_face_position(self):
        if self.num_faces:
            self.face_position_counter += 1
            if self.face_position_counter > 20:
                return True
        return False

    def get_liveness(self, camera_params, depth_frame):
        bounding_box = self.detection_result.faces[0].bounding_box
        cropped_depth_frame = self.liveness.crop_depth_frame(camera_params, depth_frame, bounding_box)
        self.window.append(cropped_depth_frame)
        if len(self.window) == 5:
            liveness_probability = self.liveness.compute_liveness_probability(self.window)
            self.livenessFlag = True
            print("liveness probability: ", liveness_probability)
            if liveness_probability > self.FACE_CONFIG.liveness_threshold:
                self.face_result.liveness = liveness_probability
            else:
                self.face_result = utils.FaceDetectionResult()

    def get_embeddings(self, image):
        image_result = self.session.get_faces([image], embeddings=True)
        return image_result

    def get_faces(self, _in_image):
        best_face_result = utils.FaceDetectionResult()
        detection_result = self.session.get_faces([_in_image], qualities=True)
        if len(detection_result.faces) > 0:
            # Extract left and right eye positions
            left_eye_x = detection_result.faces[0].landmarks.left_eye.x
            left_eye_y = detection_result.faces[0].landmarks.left_eye.y
            right_eye_x = detection_result.faces[0].landmarks.right_eye.x
            right_eye_y = detection_result.faces[0].landmarks.right_eye.y
            # find distance between eyes. This will help to determine the distance from user to kiosk
            face_size = utils.get_distance_between_two_pints(left_eye_x, left_eye_y, right_eye_x, right_eye_y)
            if face_size > self.FACE_CONFIG.face_size_threshold:
                top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
                top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
                bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
                bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)

                # add padding
                top_left_y = top_left_y - self.FACE_CONFIG.height_padding
                bottom_right_y = bottom_right_y + self.FACE_CONFIG.height_padding
                top_left_x = top_left_x - self.FACE_CONFIG.width_padding
                if top_left_x < 0:
                    top_left_x = 0
                bottom_right_x = bottom_right_x + self.FACE_CONFIG.width_padding

                cropped_face = utils.crop_face(top_left_x, top_left_y,
                                               bottom_right_x, bottom_right_y,
                                               _in_image)
                best_face_result.faces = len(detection_result.faces)
                best_face_result.faces = len(detection_result.faces)
                best_face_result.quality = detection_result.faces[0].quality
                best_face_result.acceptability = detection_result.faces[0].acceptability
                best_face_result.liveness = 1
                best_face_result.face_size = face_size
                best_face_result.box_height = detection_result.faces[0].bounding_box.height()
                best_face_result.box_width = detection_result.faces[0].bounding_box.width()
                best_face_result.box_x = int(detection_result.faces[0].bounding_box.top_left.x)
                best_face_result.box_y = int(detection_result.faces[0].bounding_box.top_left.y)
                best_face_result.base64 = utils.convert_image_to_base64(cropped_face)
                best_face_result.top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
                best_face_result.top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
                best_face_result.bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
                best_face_result.bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)
                best_face_result.frame_height = _in_image.shape[0]
                best_face_result.frame_width = _in_image.shape[1]

        best_face_result.time = utils.get_datetime()
        return best_face_result
