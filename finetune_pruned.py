# Create a new script: finetune_pruned.py
from ultralytics import YOLO
import torch

# Load the structurally changed model
# Note: We load it as a generic PyTorch model first
pruned_pytorch_model = torch.load("yolov8n_pruned.pt")

# Wrap it back into YOLO for training
model = YOLO("yolov8n.pt") # Load a dummy to get the wrapper
model.model = pruned_pytorch_model # Swap the brains

# Fine-tune
print("Fine-tuning to recover accuracy...")
model.train(data="coco128.yaml", epochs=50, imgsz=640, lr0=0.001)

# Export to ONNX for your FPGA
model.export(format="onnx", opset=12)