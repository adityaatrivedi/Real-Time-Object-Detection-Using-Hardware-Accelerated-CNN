from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import mobilenetv3_modules as mb

# --- MONKEY PATCH START ---
# We force the YOLO library (tasks.py) to recognize your custom classes
# by adding them directly to its internal dictionary.
print("Injecting MobileNetV3 modules into YOLO...")
tasks.ConvBNActivation = mb.ConvBNActivation
tasks.InvertedResidual = mb.InvertedResidual
# --- MONKEY PATCH END ---

# Now when you load the model, the parser inside 'tasks.py' will find them!
model = YOLO("yolov8-mobilenetv3.yaml")

# Train
# Note: 'coco8.yaml' is a tiny dataset for debugging. 
# Use 'coco128.yaml' or your own data for real training.
model.train(data="coco8.yaml", epochs=100)