# Uses ticcmd to send and receive data from the Tic over USB.
# Works with either Python 2 or Python 3.
#
# NOTE: The Tic's control mode must be "Serial / I2C / USB".

import subprocess
from read_config_file import get_actuator_settings
import time
import yaml
import csv
import os

Columns = ['Date & Time', 'Camera Status', 'Forward limit active', 'Reverse limit active',
           'VIN voltage', 'Operation state', 'Energized', 'Last motor driver error',
           'Current velocity', 'Acting target position']

file_path = 'C:/Logs/EMVISION_API/Actuator/'


class RingBuffer:
    def __init__(self, size):
        self.data = [0 for i in range(size)]

    def __len__(self):
        return len(self.data)

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data

    def reset(self):
        self.data = [600 for i in range(len(self))]

    def average(self):
        result = 0
        for i in range(len(self)):
            result += self.data[i]
        return result / len(self)


class Actuator:
    def __init__(self):
        self.config = get_actuator_settings()
        print(self.config)
        self.y_filter = RingBuffer(self.config.filter_size)
        self.face_result = None
        self.top_limit = self.config.top_limit
        self.bottom_limit = self.config.bottom_limit
        self.home = self.config.home
        self.steps = self.config.steps_interval
        self.ki_up = self.config.ki_up
        self.ki_down = self.config.ki_down
        self.motor_position = self.home
        self.counter = 0
        self.home_counter = 0
        # status = yaml.safe_load(self.actuator_command('-s', '--full'))
        self.position = self.home
        # self.position = status['Current position']
        # self.first_time = True
        self.new_target = self.config.top_limit

    @staticmethod
    def actuator_command(*args):
        return subprocess.check_output(['ticcmd'] + list(args))

    def move_actuator_to(self, position):
        if self.bottom_limit > position > self.top_limit:
            self.actuator_command('--exit-safe-start', '--position', str(position))

    def handle_actuator(self, face_result):
        res = False
        if face_result.faces > 0:
            face_center_y = face_result.box_y + face_result.box_width
            self.y_filter.append(face_center_y)
            diff = int((self.y_filter.average() / face_result.frame_height) * 100)
            if diff > 55:
                self.counter = 0
                if self.motor_position > self.top_limit:
                    self.motor_position = self.motor_position + int(self.steps * abs(diff) * self.ki_down)
            elif diff < 45:
                self.counter = 0
                if self.motor_position < self.bottom_limit:
                    self.motor_position = self.motor_position - int(self.steps * abs(diff) * self.ki_up)
            else:
                self.counter += 1
                if self.counter > 5:
                    res = True
            self.move_actuator_to(int(self.motor_position))
        else:
            self.home_counter += 1
            res = False
            self.y_filter.reset()
            self.motor_position = self.home
            if self.home_counter > self.config.frames_to_wait:
                self.move_actuator_to(self.motor_position)
                self.home_counter = 0
        return res

    def exercise_actuator(self):
        if self.position == self.bottom_limit:
            self.new_target = self.top_limit
        if self.position == self.top_limit:
            self.new_target = self.bottom_limit
        elif self.position == self.home:
            self.new_target = self.top_limit
        else:
            self.new_target = self.home

        self.actuator_command('--exit-safe-start', '--position', str(self.new_target))
        if self.new_target == self.bottom_limit:
            time.sleep(self.config.rest_time_in_sec + 2)
        else:
            time.sleep(self.config.rest_time_in_sec + 1)

        self.position = self.new_target

        # return status
        return


def write_dicts_to_csv(polulu_status, _path, first_time):
    if not os.path.exists(_path):
        os.makedirs(_path)
    field_names = polulu_status.keys()
    with open(_path + '/actuator_data.csv', 'a', newline='') as csv_file_object:
        writer = csv.DictWriter(csv_file_object, fieldnames=field_names)
        if first_time:
            writer.writeheader()
            first_time = False
        writer.writerow(polulu_status)
        return first_time
