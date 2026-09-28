from multiprocessing import freeze_support

from ultralytics import YOLO


def main():
    model = YOLO("D:\\yang\\Downloads\\ultralytics-main\\ultralytics-main\\runs\\detect\\train2\\weights\\last.pt")
    model.train(
        data="D:/训练数据/yolodataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=12,
        workers=4,
        patience=5,
        save=True,
        val=True,
        warmup_epochs=5,
        warmup_momentum=0.8,
        cos_lr=True,
        resume=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
