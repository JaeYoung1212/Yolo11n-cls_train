from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model (build from scratch)
    model = YOLO("yolo11n-cls.yaml")

    # Train the model with augmentation
    results = model.train(
        data = r"C:\Users\limjy\AutoDrive\TS_TL_Classification\AutoDrive_TS_TL_main\yolo11_stop_yield\Train_dataset",
        epochs=100,
        imgsz=640,
        patience=10,
        batch=6,
        device=0,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        erasing=0.1,
    )