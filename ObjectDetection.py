"""
Welcome to CARLA manual control.
STARTING in a moment...
"""

from __future__ import print_function

import argparse
import logging
import random
import time
import datetime

###### 
import cv2   #for image
import numpy as np 
import os     # for changing dir
import scipy.io as io  ## for saving image file 
import threading
print(os.getcwd())     #print current dir
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


####CARLA 
from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

####CARLA %

i=0
WINDOW_WIDTH = 270 
WINDOW_HEIGHT = 220
MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

t1 = None

net = cv2.dnn.readNet(".\\Models\\yolov4.weights", ".\\Models\\yolov4.cfg")
classes = []
with open("./Models/yoloclasses.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(150, 255, size=(len(classes), 3))

print("Object Detection model Loaded ... ")

def make_carla_settings(args):                                       ## normal fucn in python
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=False,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=30,
        NumberOfPedestrians=30,
        WeatherId=random.choice([1, 3, 7, 8, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')                           # Set RGB Camera
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)            #
    camera0.set_position(1.0, 0.0, 2.0)                            #
    camera0.set_rotation(0.0, 0.0, 0.0)                           #
    settings.add_sensor(camera0)                                   # Adding camera in Carla
    #camera1 = sensor.Camera('CameraDepth', PostProcessing='Depth')
    #camera1.set_image_size(MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT)
    #camera1.set_position(2.0, 0.0, 1.4)
    #camera1.set_rotation(0.0, 0.0, 0.0)
    #settings.add_sensor(camera1)
    return settings                     ##return settings

class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

main_image = None

class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._timer = None
        self._display = None
        self._mini_view_image1 = None
        self._enable_autopilot = True
        self._map_view = None
        self._is_on_reverse = False
        self._display_map = args.map
        self._city_name = None
        self._map = None
        self._map_shape = None
        self._map_view = None
        self._position = None
        self._agent_positions = None
        self._i = 0

    ##EXECUTE FUNCTION IS MAIN LOOP 
    def execute(self):
        self._on_new_episode()
        while True:
            self._on_loop()
            self._i += 1
    
    def _on_new_episode(self):
        self._carla_settings.randomize_seeds()
        self._carla_settings.randomize_weather()
        scene = self.client.load_settings(self._carla_settings)
        if self._display_map:
            self._city_name = scene.map_name
        number_of_player_starts = len(scene.player_start_spots)
        player_start = np.random.randint(number_of_player_starts)
        print('Starting new episode...')
        self.client.start_episode(player_start)
        self._timer = Timer()
        self._is_on_reverse = False

    def _on_loop(self):    # called in execute func
        self._timer.tick()
        measurements, sensor_data = self.client.read_data()
        global main_image
        main_image = sensor_data.get('CameraRGB', None)
        if self._i == 0 :
            global t1
            t1 = threading.Thread(target=show_proccessed_image, name='t1')
            t1.start()
        if self._timer.elapsed_seconds_since_lap() > 1.0:
            if self._city_name is not None:
                # Function to get car position on map.
                map_position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                # Function to get orientation of the road car is in.
                lane_orientation = self._map.get_lane_orientation([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])
                    
                self._print_player_measurements_map(
                    measurements.player_measurements,
                    map_position,
                    lane_orientation)
            
            else:
                self._print_player_measurements(measurements.player_measurements)

            self._timer.lap()
        if self._city_name is not None:
            self._position = self._map.convert_to_pixel([
                measurements.player_measurements.transform.location.x,
                measurements.player_measurements.transform.location.y,
                measurements.player_measurements.transform.location.z])
            self._agent_positions = measurements.non_player_agents
        # if self._i < 0 :
        #     global t1
        #     t1.join()
        
        self.client.send_control(measurements.player_measurements.autopilot_control)
 
    def _print_player_measurements_map(self, player_measurements, map_position, lane_orientation):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += 'Map Position ({map_x:.1f},{map_y:.1f}) '
        message += 'Lane Orientation ({ori_x:.1f},{ori_y:.1f}) '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            map_x=map_position[0],
            map_y=map_position[1],
            ori_x=lane_orientation[0],
            ori_y=lane_orientation[1],
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)
        self._velocity = player_measurements.forward_speed *3.6

    def _print_player_measurements(self, player_measurements):
        message = 'Step {step} ({fps:.1f} FPS): '
        message += '{speed:.2f} km/h, '
        message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road'
        message = message.format(
            step=self._timer.step,
            fps=self._timer.ticks_per_second(),
            speed=player_measurements.forward_speed * 3.6,
            other_lane=100 * player_measurements.intersection_otherlane,
            offroad=100 * player_measurements.intersection_offroad)
        print_over_same_line(message)

def show_proccessed_image():
    while True :
        global main_image
        if main_image is not None:
            array = image_converter.to_rgb_array(main_image)
            img = object_detection(array)
            screen = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2.imshow("Object Detection", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                CarlaGame._i = -1000    
                break
        else :
            # print("no object")
            pass

def object_detection(img) :
    # print("object detecting")
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
    # start_time = time.time()
    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(time.time() - start_time)
    # time.sleep(1.5)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x, y), font, 1, color, 2)
    return img
    # cv2.imshow("Object Detection", img)
    # cv2.waitKey(2000)

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level, a lower level makes the simulation run considerably faster')
    argparser.add_argument(
        '-m', '--map',
        action='store_true',
        help='plot the map of the current city')
    args = argparser.parse_args()
    logging.info('listening to server %s:%s', args.host, args.port)
    while True:
        try:
            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break
        except TCPConnectionError as error:
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

##################################################################################   Bus bhahut hoga aaj ke liye.....
