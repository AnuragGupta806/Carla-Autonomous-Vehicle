# Carla-Autonomous-Vehicle
## Team Members
1. Anurag Gupta
2. Amit Gupta
3. Mohd. Sahil
4. Ankur Pratap Singh

## Phase 1:Object Detection
First part of our project involved detecting real world objects in carla simulated environment. For this, we took help of a well known detection model called YOLO4(You Only Look once) because of its great tradeoff between Speed and Accuracy.

- YOLO4 weights file can be downloaded from this [Link](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and placed in the models folder

- Youtube video of its Live Demo can be found at this [Link](https://www.youtube.com/watch?v=Vct5sLkOILU&t=56s)

## Phase 2: Road Detection using Semantic Segmentation
In this phase, we trained a road segmentation CNN model on the dataset of semantic images from carla to to detect road pixels from the camera image. After that we used Random sample consensus(RANSAC), that estimates the road plane from given set of points by taking multiple sets if 3 random points and passing a plane through it.

- Road segmentation model file is present in 'Models' folder
- Youtube video for the same can be found at this [Link]
