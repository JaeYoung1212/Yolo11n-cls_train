from ultralytics import YOLO

# Load the exported OpenVINO model
model = YOLO(r"runs\classify\train7\weights\best_openvino_model")


results = model(r"yolo11_stop_yield/real_test")

for r in results:
    print(r.path)   
    print(r.probs)  