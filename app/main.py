from fastapi import FastAPI
import uvicorn
from camera_module.camera import CameraThread
from router.routes import Emvision_API
from utilities.utils import get_software_info
from read_config_file import *
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import signal
import sys

get_raw_config_data()

app = FastAPI(
    title="Emvision API",
    description="Emvision API offers state-of-the-art face detection technology.",
    version=get_software_info()
    )

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def setup_logs():
    log_config = get_log_config()
    logger = logging.getLogger()
    if log_config.level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_config.level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    log_folder = "EMVISION_API"
    log_file_name = log_folder + '/EMVISION_API.DBG'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)d %(levelname)-8s %(message)s',
                                  datefmt='%Y%m%d %H:%M:%S')

    timed_rotate_fh = TimedRotatingFileHandler(filename=log_file_name,
                                               when='midnight',
                                               backupCount=log_config.backup_in_days)
    logger.addHandler(timed_rotate_fh)
    timed_rotate_fh.setFormatter(formatter)


def start_uvicorn_server():
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    del uvicorn_log_config["loggers"]
    uvicorn.run(app, host=get_host_url(), port=get_port_num(), log_config=uvicorn_log_config)


app.include_router(Emvision_API)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    setup_logs()
    # create threads
    camera_thread1 = CameraThread(camera_config=get_camera_settings())
    camera_thread1.daemon = True
    camera_thread1.start()
    start_uvicorn_server()  # run as thread later
    camera_thread1.stop = True
    print("waiting for camera thread")
    camera_thread1.join()
    print("everything finished")
