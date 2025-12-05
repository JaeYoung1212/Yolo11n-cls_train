from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/train8/weights/best.pt") 


results = model("yolo11_stop_yield/real_test")  # predict on all images in folder


for r in results:
    print(r.path)   
    print(r.probs)  