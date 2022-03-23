import cv2
from paravision.recognition import Session, Engine
import threading
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

color = (255, 0, 0)
thickness = 2

class GPUThread(threading.Thread):
    def __init__(self, session):
        threading.Thread.__init__(self)
        self.session = session
    def run(self):
        #cuda.init()
        #self.dev = cuda.Device(0)
        #self.ctx = self.dev.make_context()
        img = cv2.imread("1.jpg")
        session = Session(engine=Engine.TENSORRT)
        detection_result = session.get_faces([img], qualities=True)
        if len(detection_result.faces) > 0:
            top_left_x = int(detection_result.faces[0].bounding_box.top_left.x)
            top_left_y = int(detection_result.faces[0].bounding_box.top_left.y)
            bottom_right_x = int(detection_result.faces[0].bounding_box.bottom_right.x)
            bottom_right_y = int(detection_result.faces[0].bounding_box.bottom_right.y)
            print(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            img = cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
        print("ran successfully")
        time.sleep(300)

        # self.array_gpu = cuda.mem_alloc(some_array.nbytes)
        # cuda.memcpy_htod(self.array_gpu, some_array)

        # test_kernel(self.array_gpu)
        # print "successful exit from thread %d" % self.number
        # self.ctx.pop()

        # del self.array_gpu
        # del self.ctx

if __name__ == "__main__":
    gpu_thread = GPUThread(1)
    gpu_thread.start()




