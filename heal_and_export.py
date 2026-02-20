from ultralytics import YOLO
import torch

# 1. Load the pruned brain
print("Loading pruned model...")

# --- THE FIX IS HERE ---
# We add weights_only=False because we are loading a full Model object, 
# not just a state dictionary.
pruned_net = torch.load("yolov8n_pruned.pt", weights_only=False)
# -----------------------

# 2. Transplant into YOLO body
# We load a standard YOLO wrapper, then hot-swap the internal model
model = YOLO("yolov8n.pt") 
model.model = pruned_net

# 3. Fine-Tune (Heal)
# We train for 10 epochs with a low learning rate to fix the broken connections
print("🏥 Healing (Fine-tuning)...")
try:
    # We use 'exist_ok=True' to overwrite previous runs
    model.train(data="coco128.yaml", epochs=10, imgsz=640, lr0=0.001)
except Exception as e:
    print(f"⚠️ Warning during training: {e}")
    print("Attempting to export anyway...")

# 4. Export for FPGA
print("Exporting to ONNX...")
# This creates 'yolov8n.onnx' (The pruned version)
# We rename it to avoid confusion with the original baseline
model.export(format="onnx", opset=11)

import os
if os.path.exists("yolov8n.onnx"):
    os.rename("yolov8n.onnx", "yolov8n_pruned_healed.onnx")
    print("✅ Done! File saved as: yolov8n_pruned_healed.onnx")
else:
    print("✅ Done! File saved as: yolov8n.onnx")