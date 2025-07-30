from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO('yolov8n.yaml')
    model.train(
        data='D:/训练数据/yolo_dataset/data.yaml',
        epochs=60,
        imgsz=640,
        batch=12,
        workers=4,
        patience=5,
        save=True,        # 每轮保存
        val=True,         # 每轮验证
        device=0,
        warmup_epochs=5,
        cos_lr=True,
    )

if __name__ == '__main__':
    freeze_support()
    main()