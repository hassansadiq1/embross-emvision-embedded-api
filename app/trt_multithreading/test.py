import threading
import time
from trtInference import TRTInference, trt

exitFlag = 0

class myThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args
   def run(self):
      print ("Starting " + self.args[0])
      self.func(*self.args)
      print ("Exiting " + self.args[0])

if __name__ == '__main__':
    # Create new threads
    '''
    format thread:
        - func: function names, function that we wished to use
        - arguments: arguments that will be used for the func's arguments
    '''

    trt_engine_path = 'mnist.trt'

    max_batch_size = 1
    trt_inference_wrapper = TRTInference(trt_engine_path,
        trt_engine_datatype=trt.DataType.FLOAT,
        batch_size=max_batch_size)

    # Get TensorRT SSD model output
    input_img_path = '/usr/src/tensorrt/data/mnist/3.pgm'

    thread1 = myThread(trt_inference_wrapper.infer, [input_img_path])

    # Start new Threads
    thread1.start()
    thread1.join()
    trt_inference_wrapper.destory()
    print ("Exiting Main Thread")