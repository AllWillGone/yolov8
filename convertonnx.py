from ultralytics import YOLO

model = YOLO(r"D:\yang\Downloads\ultralytics-main\ultralytics-main\runs\detect\train2\weights\best.pt")

model.export(format="onnx")
