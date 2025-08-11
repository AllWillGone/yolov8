import torch
from ultralytics import YOLO
import cv2

model=YOLO('D:\\yang\\Downloads\\ultralytics-main\\ultralytics-main\\runs\\detect\\train2\\weights\\best.pt')

result=model.predict(source='D:\\yang\\Downloads\\ultralytics-main\\ultralytics-main\\test.jpg',save=True)
