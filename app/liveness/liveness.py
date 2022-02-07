from paravision.liveness.session import Liveness
from paravision.recognition import Session, Engine
import pyrealsense2 as rs
from paravision.liveness import CameraParams
import numpy as np
import cv2

session = Session(engine=Engine.TENSORRT)
liveness = Liveness(settings={"max_batch_size": 1})

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break

if not found_rgb:
    print("The demo requires Depth camera with Color sensor")

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

depth_stream = profile.get_stream(rs.stream.depth)
color_stream = profile.get_stream(rs.stream.color)
depth_profile = depth_stream.as_video_stream_profile()
color_profile = color_stream.as_video_stream_profile()
depth_intr = depth_profile.get_intrinsics()
color_intr = color_profile.get_intrinsics()
color_to_depth_extr = color_profile.get_extrinsics_to(depth_stream)

camera_params = CameraParams(depth_intr, color_intr, color_to_depth_extr)

window = []
for i in range(20):
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(str(i) + "_rgb.jpg", color_image)
    cv2.imwrite(str(i) + "_depth.jpg", depth_image)
    # Extract face from rgb image
    face = session.get_faces([color_image], qualities=True)
    if len(face.faces) > 0:
        print("Face detected")
        bounding_box = face.faces[0].bounding_box
        top_left_x = int(face.faces[0].bounding_box.top_left.x)
        top_left_y = int(face.faces[0].bounding_box.top_left.y)
        bottom_right_x = int(face.faces[0].bounding_box.bottom_right.x)
        bottom_right_y = int(face.faces[0].bounding_box.bottom_right.y)
        crop_rgb = color_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        crop_depth = depth_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        cv2.imwrite(str(i) + "_crop_rgb.jpg", crop_rgb)
        cv2.imwrite(str(i) + "_crop_depth.jpg", crop_depth)
        # Get depth crop to be fed to liveness model
        cropped_depth_frame = liveness.crop_depth_frame(camera_params, depth_image, bounding_box)
        window.append(cropped_depth_frame)
        if len(window) == 5:
            break

# Stop streaming
pipeline.stop()
liveness_probability = liveness.compute_liveness_probability(window)
print("liveness probability: ", liveness_probability)

