from ultralytics import YOLO

# 1. Load the standard YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# 2. Export to ONNX format
# We use opset=11 or 12 because FPGA tools often prefer slightly older opsets for compatibility.
model.export(format='onnx', opset=11, imgsz=640)

print("Export complete. File saved as 'yolov8n.onnx'")