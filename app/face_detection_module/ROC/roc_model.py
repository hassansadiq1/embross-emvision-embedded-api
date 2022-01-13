import warnings

import app.utilities.utils as utils
from python3.roc import *
import cv2
from PIL import Image
import json
from numpy import interp


QUALITY_THRESHOLD = 0
CONFIDENCE_THRESHOLD = 0


class ROCFaceDetection(object):
    def __init__(self):
        roc_ensure(roc_initialize(None, None))

    def get_faces(self, _image):
        result = utils.FaceDetectionResult()
        image = convert_cv_to_roc(_image)
        probe = roc_template()
        roc_ensure(roc_represent(image,
                                 ROC_FRONTAL_DETECTION | ROC_FAST_REPRESENTATION | ROC_SPOOF,
                                 ROC_SUGGESTED_ABSOLUTE_MIN_SIZE,
                                 -1,
                                 ROC_SUGGESTED_FALSE_DETECTION_RATE,
                                 ROC_SUGGESTED_MIN_QUALITY, probe))

        if not probe.algorithm_id:
            roc_ensure("Failed to detect face in image")
            warnings.warn("No Face Detected in given duration")
        else:
            json_parsed = json.loads(probe.md)
            box_x1 = int(probe.detection.x - (probe.detection.width / 2))
            box_y1 = int(probe.detection.y - (probe.detection.height / 2))
            box_x2 = int(probe.detection.width) + box_x1
            box_y2 = int(probe.detection.height) + box_y1
            quality = (interp(json_parsed["Quality"], [-2, 2], [0, 0.9999]))
            acceptability = max(min(0.9999, probe.detection.confidence), 0)
            if quality > QUALITY_THRESHOLD and acceptability > CONFIDENCE_THRESHOLD:
                result.box_height = int(probe.detection.height)
                result.box_width = int(probe.detection.width)
                result.box_x = int(probe.detection.x)
                result.box_y = int(probe.detection.y)
                crop_img = utils.crop_face(box_x1, box_y1, box_x2, box_y2, _image)
                result.base64 = utils.convert_image_to_base64(crop_img)
                result.quality = quality
                result.acceptability = acceptability
                result.faces = 1
                if "Spoof" in json_parsed:
                    result.liveness = json_parsed["Spoof"]
        result.time = utils.get_datetime()
        delete_uint8_t_array(image.data)
        roc_ensure(roc_free_template(probe))
        return result

    def compare_faces(self, _img1_base64, _img2_base64):
        result = utils.ImageComparision()
        image1 = convert_cv_to_roc(utils.convert_base64_to_image(_img1_base64))
        image2 = convert_cv_to_roc(utils.convert_base64_to_image(_img2_base64))
        probe1 = roc_template()
        probe2 = roc_template()
        roc_ensure(roc_represent(image1,
                                 ROC_FRONTAL_DETECTION | ROC_STANDARD_REPRESENTATION,
                                 ROC_SUGGESTED_ABSOLUTE_MIN_SIZE,
                                 -1,
                                 ROC_SUGGESTED_FALSE_DETECTION_RATE,
                                 ROC_SUGGESTED_MIN_QUALITY, probe1))
        roc_ensure(roc_represent(image2,
                                 ROC_FRONTAL_DETECTION | ROC_STANDARD_REPRESENTATION,
                                 ROC_SUGGESTED_ABSOLUTE_MIN_SIZE,
                                 -1,
                                 ROC_SUGGESTED_FALSE_DETECTION_RATE,
                                 ROC_SUGGESTED_MIN_QUALITY,
                                 probe2))
        if not probe1.algorithm_id:
            roc_ensure("Failed to detect face in image1")
            warnings.warn("No Face Detected in Image 1")

        elif not probe2.algorithm_id:
            roc_ensure("Failed to detect face in image2")
            warnings.warn("No Face Detected in Image 2")
        else:
            similarity = new_roc_similarity()
            roc_ensure(roc_compare_templates(probe1, probe2, similarity))
            result.score = max(min(0.9999, roc_similarity_value(similarity)), 0)
            json_parsed1 = json.loads(probe1.md)
            json_parsed2 = json.loads(probe2.md)
            result.image1_face_quality = interp(json_parsed1["Quality"], [-2, 2], [0, 0.9999])
            result.image2_face_quality = interp(json_parsed2["Quality"], [-2, 2], [0, 0.9999])
            result.image1_number_of_faces = 1
            result.image2_number_of_faces = 1
            delete_roc_similarity(similarity)
            roc_ensure(roc_free_image(image1))
            roc_ensure(roc_free_image(image2))
            roc_ensure(roc_free_template(probe1))
            roc_ensure(roc_free_template(probe2))
        result.time = utils.get_datetime()
        return result


def convert_cv_to_roc(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pillow = Image.fromarray(img)
    bytes = 3 * image_pillow.width * image_pillow.height
    image_roc = roc_image()
    image_roc.width = image_pillow.width
    image_roc.height = image_pillow.height
    image_roc.step = 3 * image_pillow.width
    image_roc.color_space = ROC_BGR24
    image_roc.data = new_uint8_t_array(bytes + 1)  # See memmove() comment in `roc_example_flatten.py`
    memmove(image_roc.data, image_pillow.tobytes())
    roc_ensure(roc_swap_channels(image_roc))
    return image_roc
