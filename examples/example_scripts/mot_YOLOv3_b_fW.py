import cv2 as cv
import time
import requests
from tempfile import NamedTemporaryFile
from PIL import Image
import sys
import numpy as np
import glob

from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker, KFTrackerSORT, KFTracker2D
from motrackers.utils import draw_tracks

import argparse

parser = argparse.ArgumentParser(
    description='Object detections in input video using YOLOv3 trained on COCO dataset.'
)

parser.add_argument(
    '--video', '-v', type=str, default="./../video_data/cars.mp4", help='Input video path.')

parser.add_argument(
    '--output', '-o', type=str, default="./../output/prediction.avi", help='Output video path.')

parser.add_argument(
    '--weights', '-w', type=str,
    default="./../pretrained_models/yolo_weights/yolov3.weights",
    help='path to weights file of YOLOv3 (`.weights` file.)'
)

parser.add_argument(
    '--config', '-c', type=str,
    default="./../pretrained_models/yolo_weights/yolov3.cfg",
    help='path to config file of YOLOv3 (`.cfg` file.)'
)

parser.add_argument(
    '--labels', '-l', type=str,
    default="./../pretrained_models/yolo_weights/coco_names.json",
    help='path to labels file of coco dataset (`.names` file.)'
)

parser.add_argument(
    '--gpu', type=bool,
    default=False, help='Flag to use gpu to run the deep learning model. Default is `False`'
)

parser.add_argument(
    '--tracker', type=str, default='CentroidTracker',
    help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")

args = parser.parse_args()

if args.tracker == 'CentroidTracker':
    tracker = CentroidTracker(max_lost=2, tracker_output_format='mot_challenge')
elif args.tracker == 'CentroidKF_Tracker':
    tracker = CentroidKF_Tracker(max_lost=7, tracker_output_format='mot_challenge')
elif args.tracker == 'KalmanFilter':
    tracker = KFTracker2D(time_step = 1)
elif args.tracker == 'SORT':
    tracker = SORT(max_lost=5, tracker_output_format='mot_challenge', iou_threshold=0)
elif args.tracker == 'IOUTracker':
    tracker = IOUTracker(max_lost=4, iou_threshold=0.5, min_detection_confidence=0.3, max_detection_confidence=0.7,
                             tracker_output_format='mot_challenge')
else:
    raise NotImplementedError

model = YOLOv3(
    weights_path=args.weights,
    configfile_path=args.config,
    labels_path=args.labels,
    confidence_threshold=0.4,
    nms_threshold=0.2,
    draw_bboxes=True,
    use_gpu=args.gpu
)

video_path = args.video
camera_IP = '169.254.7.82'
cap = cv.VideoCapture('http://test1234:Test1234@{}/cgi-bin/mjpeg?stream=[1]'.format(camera_IP))
#cap = cv.VideoCapture(0)

if not cap:
    print("!!! Failed VideoCapture: invalid parameter!")

writer = None

framerate = 2

while True:
    time.sleep(1/framerate)

    ok, image = cap.read()

    if not ok:
        print("Cannot read the video feed.")
        break

    image = cv.resize(image, (1400, 1000))
    #print(image)
    #print(type(image))
    #print(image.shape)

    #cv.imshow("img", image)
    #time.sleep(3)
    
    bboxes, confidences, class_ids = model.detect(image)
    tracks = tracker.update(bboxes, confidences, class_ids)
    updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)
    updated_image = draw_tracks(updated_image, tracks)
    

    #cv.imshow("image", image)
    cv.imshow("image", updated_image)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if writer is None:
		    # initialize our video writer
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args.output, fourcc, 2,
            (updated_image.shape[1], updated_image.shape[0]), True)

	    # write the output frame to disk
    writer.write(updated_image)

print("[INFO] cleaning up...")
cap.release()
#writer.release()
cv.destroyAllWindows()