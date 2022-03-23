import cv2
from paravision.recognition import Session, Engine
import threading

color = (255, 0, 0)
thickness = 2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(flip_method=0))
    session = Session(engine=Engine.TENSORRT)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            detection_result = session.get_faces([img], qualities=True)
            if len(detection_result.faces) > 0:
                top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
                top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
                bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
                bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)
                img = cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
            # cv2.imshow("CSI Camera", img)

            # keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            # if keyCode == 27:
                # break
        cap.release()
        print("ran successfully")
        # cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    x = threading.Thread(target=show_camera)
    x.start()
