import pycuda
import cv2
import pycuda.driver as cuda
from paravision.recognition import Session, Engine
from pycuda.compiler import SourceModule
import threading
import numpy
import time

class GPUThread(threading.Thread):
    def __init__(self, number, some_array):
        threading.Thread.__init__(self)

        self.number = number
        self.some_array = some_array

    def run(self):
        img = cv2.imread("1.jpg")
        session = Session(engine=Engine.TENSORRT)
        detection_result = session.get_faces([img], qualities=True)
        print(self.number, " ran successfully: ",detection_result.faces[0].bounding_box.top_left.y)
        del session
        time.sleep(5)

cuda.init()
some_array = numpy.ones((1,512), dtype=numpy.float32)
num = 4

gpu_thread_list = []
for i in range(num):
    gpu_thread = GPUThread(i, some_array)
    gpu_thread.start()
    gpu_thread_list.append(gpu_thread)
