from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs\classify\train7\weights\best.pt")

# Export to OpenVINO
model.export(format="openvino")