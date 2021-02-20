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
import tensorflow as tf
import matplotlib.pyplot as plt
###### 
import cv2   #for image
import numpy as np 
import os     # for changing dir
import scipy.io as io  ## for saving image file 
import threading
import math
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
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 280
MINI_WINDOW_WIDTH = 300
MINI_WINDOW_HEIGHT = 150

t1 = None
t2 = None

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
        NumberOfVehicles=15,
        NumberOfPedestrians=15,
        WeatherId=random.choice([1, 2, 7, 8, 9, 14]),
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    camera0 = sensor.Camera('CameraRGB')                           # Set RGB Camera
    camera0.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT)            #
    camera0.set_position(1.0, 0.0, 2.0)                            #
    camera0.set_rotation(0.0, 0.0, 0.0)                           # pitch,yaw,roll
    settings.add_sensor(camera0)                                   # Adding camera in Carla

    camera_depth = sensor.Camera('Depth_Camera', PostProcessing='Depth')
    camera_depth.set(FOV=90.0)
    camera_depth.set_image_size(WINDOW_WIDTH, WINDOW_HEIGHT) 
    camera_depth.set_position(1.0, 0.0, 2.0)                            #
    camera_depth.set_rotation(0.0, 0.0, 0.0)   

    settings.add_sensor(camera_depth)

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
depth_image = None
shutdown = False

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
        self._model = tf.keras.models.load_model(args.model)
        self._i = 0
        self._f = WINDOW_WIDTH/(2 * math.tan(math.pi/4))
        self._uc = WINDOW_WIDTH/2
        self._uv = WINDOW_HEIGHT/2

    ##EXECUTE FUNCTION IS MAIN LOOP 
    def execute(self):
        self._on_new_episode()
        while True:
            self._on_loop()
            self._i += 1
            # break
    
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

        global main_image, depth_image
        main_image = sensor_data.get('CameraRGB', None)
        depth_image = sensor_data.get('Depth_Camera', None)

        if self._i == 0 :
            global t1
            t1 = threading.Thread(target=self._RANSAC, name='t1')
            t1.start()
            global t2
            t2 = threading.Thread(target=self._show_proccessed_image, name='t1')
            t2.start()
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

    def _xy_from_depth(self, depth):
        """
        Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.
        """
        ### START CODE HERE ### (≈ 7 lines in total)
        # Get the shape of the depth tensor
        print(depth.shape)
        H, W = np.shape(depth)
        # Grab required parameters from the K matrix
        f = self._f
        c_u = self._uc
        c_v = self._uv
        # Generate a grid of coordinates corresponding to the shape of the depth map
        x = np.zeros((H, W))
        y = np.zeros((H, W))
        # Compute x and y coordinates
        for i in range(H):
            for j in range(W):
                x[i, j] = ((j+1 - c_u)*depth[i, j]) / f
                y[i, j] = ((i+1 - c_v)*depth[i, j]) / f
        ### END CODE HERE ###
        return x, y

    def _compute_plane(self, xyz):
        """
        Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
        Arguments:
        """
        ctr = xyz.mean(axis=1)
        normalized = xyz - ctr[:, np.newaxis]
        M = np.dot(normalized, normalized.T)

        p = np.linalg.svd(M)[0][:, -1]
        d = np.matmul(p, ctr)

        p = np.append(p, -d)

        # Correct plane
        # p = [0.0, 1.0, 0.0, -1.5]
        return p

    def _dist_to_plane(self, plane, x, y, z):
        """
        Computes distance between points provided by their x, and y, z coordinates
        and a plane in the form ax+by+cz+d = 0
        """
        a, b, c, d = plane
    
        return (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

    def _ransac_plane_fit(self, xyz_data):
        """
        Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
        using ransac for outlier rejection.
        """
        ### START CODE HERE ### (≈ 23 lines in total)
    
        # Set thresholds:
        num_itr = 100  # RANSAC maximum number of iterations
        min_num_inliers = xyz_data.shape[1] / 2  # RANSAC minimum number of inliers
        distance_threshold = 0.008  # Maximum distance from point to plane for point to be considered inlier
        largest_number_of_inliers = 0
        largest_inlier_set_indexes = 0
    
        for i in range(num_itr):
            # Step 1: Choose a minimum of 3 points from xyz_data at random.
            indexes = np.random.choice(xyz_data.shape[1], 3, replace = False)
            pt1 = xyz_data[:, indexes[0]]
            pt2 = xyz_data[:, indexes[1]]
            pt3 = xyz_data[:, indexes[2]]
            pts = np.stack((pt1, pt2, pt3))
            pts = xyz_data[:, indexes]
            # print(pts.shape)
        
            # Step 2: Compute plane model
            p = self._compute_plane(pts)
        
            # Step 3: Find number of inliers
            distance = self._dist_to_plane(p, xyz_data[0, :].T, xyz_data[1, :].T, xyz_data[2, :].T)
            number_of_inliers = len(distance[distance < distance_threshold])
        
            # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
            if number_of_inliers > largest_number_of_inliers:
                largest_number_of_inliers = number_of_inliers
                largest_inlier_set_indexes = np.where(distance < distance_threshold)[0]

            # Step 5: Check if stopping criterion is satisfied and break.
            if (number_of_inliers > min_num_inliers):
                break

        # Step 6: Recompute the model parameters using largest inlier set.
        output_plane = self._compute_plane(xyz_data[:, largest_inlier_set_indexes])

        ### END CODE HERE ###
    
        return output_plane

    def _RANSAC(self) :
        """
        Show sementic segmentation image
        """
        while True:
            global shutdown
            if shutdown :
                break
            global depth_image, main_image
            if depth_image is not None and main_image is not None:
                try:
                    array = image_converter.depth_to_array(depth_image)
                except ValueError as e:
                    continue
                z = array*1000
                x, y = self._xy_from_depth(z)

                camera_image = image_converter.to_rgb_array(main_image)

                array = np.expand_dims(camera_image, 0)
                t = time.time()
                my_preds = self._model.predict(array)
                my_preds = my_preds.flatten()
                my_preds = np.array([1 if i >= 0.5 else 0 for i in my_preds])
                print("Segmentation : Time = " + str(time.time() - t))
                road_mask = my_preds.reshape(280, 400)
                road_mask = road_mask.astype(np.float32)

                x_ground = x[road_mask == 1]
                y_ground = y[road_mask == 1]
                z_ground = z[road_mask == 1]
                xyz_ground = np.stack((x_ground, y_ground, z_ground))

                p_final = self._ransac_plane_fit(xyz_ground)
                print('Ground Plane: ' + str(p_final))

                dist = np.abs(self._dist_to_plane(p_final, x, y, z))

                ground_mask = np.zeros(dist.shape)

                ground_mask[dist < 0.05] = 1
                ground_mask[dist > 0.05] = 0

                blue_channel = np.zeros(ground_mask.shape, dtype=ground_mask.dtype)
                green_channel = np.zeros(ground_mask.shape, dtype=ground_mask.dtype)

                ground_mask = cv2.merge((blue_channel, green_channel, ground_mask))

                camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
                # camera_image = camera_image.astype(np.uint8)*255
                # cv2.imshow("camera", camera_image)
                # cv2.imshow("ml model", road_mask)
                # cv2.imshow("mask", ground_mask)

                ground_mask = ground_mask.astype(np.uint8)*255
                rows, cols, channels = ground_mask.shape
                roi = camera_image[0:rows, 0:cols]
                img2gray = cv2.cvtColor(ground_mask, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 20, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)
                img1_bg = cv2.bitwise_and(roi, roi, mask = mask)
                img2_fg = cv2.bitwise_and(ground_mask, ground_mask, mask = mask_inv)
                out_img = cv2.add(img1_bg,img2_fg)
                camera_image[0:rows, 0:cols ] = out_img

                cv2.imshow("road detection", camera_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # break
            else :
                print("no image")

    def _show_proccessed_image(self):
        while True :
            global shutdown
            if shutdown :
                break
            global main_image
            if main_image is not None:
                array = image_converter.to_rgb_array(main_image)
                img = self._object_detection(array)
                screen = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                cv2.imshow("Object Detection", screen)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else :
             print("no object")

    def _object_detection(self, img) :
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
        start_time = time.time()
        net.setInput(blob)
        outs = net.forward(output_layers)
        print("Object Detection : Time = " + str(time.time() - start_time))
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
        '-M', '--model',
        help='Machine learing model',
        required=True)
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
        shutdown = True
        t1.join()
        t2.join()
        print('\nCancelled by user. Bye!')

##################################################################################   Bus bhahut hoga aaj ke liye.....
