from click.core import batch

from ultralytics import YOLO
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

model = YOLO("yolov8n.pt") # pass any model type

