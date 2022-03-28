import json
from utilities.utils import CameraSettings, LogConfig, CroppedFaceSettings, ActuatorSettings


raw_json_data: dict
file_path = './config.json'


def get_raw_config_data():
    try:
        with open(file_path) as f:
            global raw_json_data
            raw_json_data = json.load(f)
    except Exception:
        raise Exception("couldn't read config.json file")


def get_camera_settings():
    camera_config = CameraSettings()
    if raw_json_data is not None:
        camera_config.id = raw_json_data["camera"].get("index")
        camera_config.name = raw_json_data["camera"].get("name")
        camera_config.frames_per_sec = raw_json_data["camera"].get("frame_rate")
        camera_config.frame_height = raw_json_data["camera"].get("height")
        camera_config.frame_width = raw_json_data["camera"].get("width")
        camera_config.brightness = raw_json_data["camera"].get("brightness")
        camera_config.contrast = raw_json_data["camera"].get("contrast")
        camera_config.hue = raw_json_data["camera"].get("hue")
        camera_config.saturation = raw_json_data["camera"].get("saturation")
        camera_config.sharpness = raw_json_data["camera"].get("sharpness")
        camera_config.gamma = raw_json_data["camera"].get("gamma")
        camera_config.backlight = raw_json_data["camera"].get("backlight")
        camera_config.rotation = raw_json_data["camera"].get("rotation_cw")
        camera_config.liveness = raw_json_data["camera"].get("liveness")
    else:
        raise Exception("config.jason file is None")

    return camera_config


def get_econ_camera_settings():
    econ_config = CameraSettings()
    if raw_json_data is not None:
        econ_config.id = raw_json_data["E-Con"].get("index")
        econ_config.name = raw_json_data["E-Con"].get("name")
        econ_config.frame_width = raw_json_data["E-Con"].get("width")
        econ_config.frame_height = raw_json_data["E-Con"].get("height")
        econ_config.frames_per_sec = raw_json_data["E-Con"].get("frame_rate")
        econ_config.rotation = raw_json_data["E-Con"].get("rotation_cw")
    else:
        raise Exception("config.jason file is None")
    return econ_config


def get_host_url():
    return raw_json_data["server"].get("host_url")


def get_port_num():
    return raw_json_data["server"].get("port_num")


def get_face_detection_sdk():
    get_raw_config_data()
    return raw_json_data["face_detection"].get("sdk")


def get_face_settings():
    get_raw_config_data()
    cropped_face_config = CroppedFaceSettings()
    cropped_face_config.face_size_threshold = raw_json_data["face_detection"].get("FACE_SIZE_THRESHOLD")
    cropped_face_config.height_padding = raw_json_data["face_detection"].get("HEIGHT_PADDING")
    cropped_face_config.width_padding = raw_json_data["face_detection"].get("WIDTH_PADDING")
    cropped_face_config.liveness_window = raw_json_data["face_detection"].get("LIVENESS_WINDOW")
    cropped_face_config.liveness_threshold = raw_json_data["face_detection"].get("LIVENESS_THRESHOLD")
    return cropped_face_config


def get_authentication_info():
    return raw_json_data["authentication"].get("ACCESS_TOKEN_EXPIRE_MINUTES")


def get_user_db():
    return raw_json_data["USERS_DB"]


def get_log_config():
    log_configuration = LogConfig()
    log_configuration.level = raw_json_data["logging"].get("level")
    log_configuration.backup_in_days = raw_json_data["logging"].get("logs_stored_for_days")
    return log_configuration


def get_actuator_settings():
    actuator_config = ActuatorSettings()
    if raw_json_data is not None:
        actuator_config.filter_size = raw_json_data["actuator"].get("filter_size")
        actuator_config.top_limit = raw_json_data["actuator"].get("top_limit")
        actuator_config.bottom_limit = raw_json_data["actuator"].get("bottom_limit")
        actuator_config.home = raw_json_data["actuator"].get("home")
        actuator_config.steps_interval = raw_json_data["actuator"].get("steps_interval")
        actuator_config.ki_up = raw_json_data["actuator"].get("ki_up")
        actuator_config.ki_down = raw_json_data["actuator"].get("ki_down")
        actuator_config.frames_to_wait = raw_json_data["actuator"].get("frames_to_wait")
        actuator_config.rest_time_in_sec = raw_json_data["actuator"].get("rest_time_in_sec")
    else:
        raise Exception("config.jason file is None")
    return actuator_config
