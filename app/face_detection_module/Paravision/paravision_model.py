from paravision.recognition import Session, Engine
from paravision.liveness.session import Liveness
import utilities.utils as utils
from read_config_file import get_face_settings, get_camera_settings
import numpy as np
from camera_module.camera import Camera

FACE_CONFIG = get_face_settings()


class ParavisionFaceDetection(object):
    def __init__(self):
        self.session = Session(engine=Engine.TENSORRT)
        camera_config = get_camera_settings()
        self.session.get_faces([np.random.rand(camera_config.frame_width,camera_config.frame_height, 3)], qualities=True)
        self.liveness = Liveness(settings={"max_batch_size": 1})

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
            if face_size > FACE_CONFIG.face_size_threshold:
                top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
                top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
                bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
                bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)

                # add padding
                top_left_y = top_left_y - FACE_CONFIG.height_padding
                bottom_right_y = bottom_right_y + FACE_CONFIG.height_padding
                top_left_x = top_left_x - FACE_CONFIG.width_padding
                if top_left_x < 0:
                    top_left_x = 0
                bottom_right_x = bottom_right_x + FACE_CONFIG.width_padding

                cropped_face = utils.crop_face(top_left_x, top_left_y,
                                               bottom_right_x, bottom_right_y,
                                               _in_image)
                best_face_result.faces = len(detection_result.faces)
                best_face_result.faces = len(detection_result.faces)
                best_face_result.quality = detection_result.faces[0].quality
                best_face_result.acceptability = detection_result.faces[0].acceptability
                best_face_result.liveness = -1
                best_face_result.face_size = face_size
                best_face_result.box_height = detection_result.faces[0].bounding_box.height()
                best_face_result.box_width = detection_result.faces[0].bounding_box.width()
                best_face_result.box_x = int(detection_result.faces[0].bounding_box.top_left.x)
                best_face_result.box_y = int(detection_result.faces[0].bounding_box.top_left.y)
                best_face_result.base64 = utils.convert_image_to_base64(cropped_face)

        best_face_result.time = utils.get_datetime()
        return best_face_result

    def compare_faces(self, _image1, _image2):
        compared_results = utils.ImageComparision()
        image1 = utils.convert_base64_to_image(_image1)
        image2 = utils.convert_base64_to_image(_image2)
        image1_result = self.session.get_faces([image1], embeddings=True)
        image2_result = self.session.get_faces([image2], embeddings=True)
        confidence = self.session.compute_confidence(image1_result.faces[0], image2_result.faces[0])

        compared_results.score = confidence
        compared_results.image1_number_of_faces = len(image1_result.faces)
        compared_results.image2_number_of_faces = len(image1_result.faces)
        compared_results.image1_face_quality = image1_result.faces[0].quality
        compared_results.image2_face_quality = image2_result.faces[0].quality
        compared_results.time = utils.get_datetime()
        return compared_results

    def get_liveness(self):
        camera_params = Camera.get_camera_params()
        best_face_result = utils.FaceDetectionResult()
        window = []
        for i in range(FACE_CONFIG.liveness_window):
            frame, depth_frame = Camera.get_current_depth_frame()
            detection_result = self.session.get_faces([frame], qualities=True)
            if len(detection_result.faces) > 0:
                left_eye_x = detection_result.faces[0].landmarks.left_eye.x
                left_eye_y = detection_result.faces[0].landmarks.left_eye.y
                right_eye_x = detection_result.faces[0].landmarks.right_eye.x
                right_eye_y = detection_result.faces[0].landmarks.right_eye.y
                # find distance between eyes. This will help to determine the distance from user to kiosk
                face_size = utils.get_distance_between_two_pints(left_eye_x, left_eye_y, right_eye_x, right_eye_y)
                if face_size > FACE_CONFIG.face_size_threshold:
                    bounding_box = detection_result.faces[0].bounding_box
                    cropped_depth_frame = self.liveness.crop_depth_frame(camera_params, depth_frame, bounding_box)
                    window.append(cropped_depth_frame)
                    if len(window) == 5:
                        top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
                        top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
                        bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
                        bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)

                        # add padding
                        top_left_y = top_left_y - FACE_CONFIG.height_padding
                        bottom_right_y = bottom_right_y + FACE_CONFIG.height_padding
                        top_left_x = top_left_x - FACE_CONFIG.width_padding
                        if top_left_x < 0:
                            top_left_x = 0
                        bottom_right_x = bottom_right_x + FACE_CONFIG.width_padding

                        cropped_face = utils.crop_face(top_left_x, top_left_y,
                                                       bottom_right_x, bottom_right_y,
                                                       frame)
                        best_face_result.faces = len(detection_result.faces)
                        best_face_result.faces = len(detection_result.faces)
                        best_face_result.quality = detection_result.faces[0].quality
                        best_face_result.acceptability = detection_result.faces[0].acceptability
                        best_face_result.liveness = -1
                        best_face_result.face_size = face_size
                        best_face_result.box_height = detection_result.faces[0].bounding_box.height()
                        best_face_result.box_width = detection_result.faces[0].bounding_box.width()
                        best_face_result.box_x = int(detection_result.faces[0].bounding_box.top_left.x)
                        best_face_result.box_y = int(detection_result.faces[0].bounding_box.top_left.y)
                        best_face_result.base64 = utils.convert_image_to_base64(cropped_face)
                        break

        if len(window) == 5:
            liveness_probability = self.liveness.compute_liveness_probability(window)
            if liveness_probability > FACE_CONFIG.liveness_threshold:
                best_face_result.liveness = liveness_probability
            else:
                best_face_result = utils.FaceDetectionResult()

        return best_face_result
