from fastapi import APIRouter, Form, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from camera_module.camera import Camera
from videostream.video_server import generate_camera_stream
import utilities.utils as utils
from face_detection_module.face_detection import FaceDetection, get_best_shot
from read_config_file import get_port_num, get_host_url, get_authentication_info, get_camera_settings
import authentication.authenticate as security


Emvision_API = APIRouter()


# will return base64 face image.
@Emvision_API.get(
    "/camera/face",
    summary="Capture a cropped face from the camera",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: Get Detected Face\t\t\n"
                "- **Description**: This call internally uses configured face detection model to detect faces.\t\t\n"
                "-  **Supported Face Detection Models**:\t\t\n"
                "   -  Paravision Model\t\t\n"
                "-  **Parameters**: \t\t\n"
                "   -  duration = 0 returns the face detected in current captured frame.\t\t\n"
                "   -  duration > 0 returns best face quality face detected in the given duration.\t\t\n"
                "-  **Return**: FaceDetectionResult. refer to the schema or example below",
    response_model=utils.FaceDetectionResult,
    tags=["Camera"]
)
async def get_face_image(duration: int,
                   current_user: utils.User = Depends(security.get_current_active_user)
                   ):
    if Camera.get_camera_status():
        frame = Camera.get_current_frame()
        if frame is not None:
            return get_best_shot(duration)
    else:
        raise HTTPException(status_code=404, detail="Camera offline")


@Emvision_API.get(
    "/camera/image",
    summary="Capture a raw frame from the camera",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: Get Camera Image\t\t\n"
                "- **Description**: This call returns the current frame captured.\t\t\n"
                "-  **Parameters**: None\t\t\n"
                "-  **Return**: RawFrame. refer to the schema or example below",
    response_model=utils.RawFrame,
    tags=["Camera"]
)
def get_camera_image(current_user: utils.User = Depends(security.get_current_active_user)):
    if Camera.get_camera_status():
        return Camera.get_base64_image()
    else:
        raise HTTPException(status_code=404, detail="Camera offline")


@Emvision_API.get(
    "/camera/stream",
    summary="Get live steam URL of Camera",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: Get Camera Stream URL\t\t\n"
                "- **Description**: This call returns URL to view live stream.\t\t\n"
                "-  **Parameters**: None\t\t\n"
                "-  **Return**: VideoStream. refer to the schema or example below",
    response_model=utils.VideoStream,
    tags=["Camera"]
)
def get_camera_stream(current_user: utils.User = Depends(security.get_current_active_user)):
    if Camera.get_camera_status():
        name = "http://" + get_host_url() + ":" + str(get_port_num())
        return {"url": name + "/camera/stream/video_raw", "time": utils.get_datetime()}
    else:
        raise HTTPException(status_code=404, detail="Camera offline")


@Emvision_API.get(
    "/camera/stream/video_raw",
    include_in_schema=False,
    summary="Streams Camera video",
    # insert response model
    tags=["Camera"]
)
def get_camera_stream():
    if Camera.get_camera_status():
        return StreamingResponse(generate_camera_stream(), media_type="multipart/x-mixed-replace;boundary=frame")
    else:
        raise HTTPException(status_code=404, detail="Camera offline")


@Emvision_API.get(
    "/camera/list",
    summary="List all the cameras",
    response_model=utils.CameraSettings,
    description="- **Name**: Get Camera status detailed\t\t\n"
                "- **Description**: This call returns the camera status with some camera settings.\t\t\n"
                "-  **Parameters**: None\t\t\n"
                "-  **Return**: CameraSettings. refer to the schema or example below",
    tags=["Camera"]
)
async def get_camera_list():
    camera_list = get_camera_settings()
    camera_list.online = Camera.get_camera_status()
    camera_list.enabled = True
    return camera_list


@Emvision_API.post(
    "/compare/images",
    summary="Compare two face images",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: 1:1 compare\t\t\n"
                "- **Description**: This call grabs both images and does face detection on both images and compares both faces to get similarity score ..\t\t\n"
                "-  **Supported Face Detection Models**:\t\t\n"
                "   -  Paravision Model\t\t\n"
                "-  **Parameters**: \t\t\n"
                "   -  img1_base64 \t\t\n"
                "   -  img2_base64 \t\t\n"
                "-  **Return**: ImageComparision. refer to the schema or example below",
    response_model=utils.ImageComparision,
    tags=["1:1 Compare"]
)
async def compare_images(
    img1_base64: str = Form(...),
    img2_base64: str = Form(...),
    current_user: utils.User = Depends(security.get_current_active_user)
):
    return FaceDetection.compare_faces(img1_base64, img2_base64)


@Emvision_API.get(
    "/software/information",
    summary="shows required software information",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: software Information\t\t\n"
                "- **Description**: This call returns software version information.\t\t\n"
                "-  **Parameters**: None\t\t\n"
                "-  **Return**: SoftwareInfo. refer to the schema or example below",
    response_model=utils.SoftwareInfo,
    tags=["software"]
)
def get_software_info(current_user: utils.User = Depends(security.get_current_active_user)):
    sw_info = utils.SoftwareInfo()
    sw_info.version = utils.get_software_info()
    return sw_info


@Emvision_API.get(
    "/auth/me/",
    description="**MANDATORY!** - User has be authorised to access this API call.\t\t\n\n "
                "- **Name**: list authenticate users\t\t\n"
                "- **Description**: This call returns all users available.\t\t\n"
                "-  **Parameters**: None\t\t\n"
                "-  **Return**: User. refer to the schema or example below",
    response_model=utils.User,
    tags=["Authentication"]
)
async def read_users_me(
        current_user: utils.User = Depends(security.get_current_active_user)
):
    return current_user


@Emvision_API.post(
    "/auth/token",
    description="- **Name**: get authorization token\t\t\n"
                "- **Description**: This call returns token to the given user if user exists\t\t\n"
                "-  **Parameters**: User name and Password are mandatory fields\t\t\n"
                "-  **Return**: Token. refer to the schema or example below",
    response_model=utils.Token,
    tags=["Authentication"]
)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = security.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = security.timedelta(get_authentication_info())
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
