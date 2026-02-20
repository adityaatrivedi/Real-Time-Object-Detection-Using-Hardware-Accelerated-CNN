from ultralytics import YOLO

# 1. Load your custom ONNX model
# Note: The library automatically detects it's an ONNX file
print("Loading ONNX model...")
model = YOLO("yolov8n.onnx", task="detect")

# 2. Run inference on a sample image
# We use a URL so you don't need to download a file manually
print("Running detection...")
results = model("https://ultralytics.com/images/bus.jpg")

# 3. Show results
# This will save the output image with boxes drawn in 'runs/detect/predict'
results[0].save() 
print(f"Success! Check the folder 'runs/detect/predict' to see the result.")