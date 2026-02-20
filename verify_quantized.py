import time
from ultralytics import YOLO
import os

# Define model paths
original_model = "yolov8n.onnx"
quantized_model = "yolov8n_quantized.onnx"

def test_model(model_path, name):
    print(f"\n--- Testing {name} ---")
    try:
        # Load model
        # The task='detect' flag forces YOLO to treat it as a detection model
        model = YOLO(model_path, task='detect')
        
        # Run inference (using a standard image URL)
        start_time = time.time()
        results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
        end_time = time.time()
        
        # Check results
        boxes = results[0].boxes
        print(f"✅ Status: Success")
        print(f"⏱️ Inference Time: {(end_time - start_time):.4f}s")
        print(f"📦 Objects Detected: {len(boxes)}")
        
        # Save result to verify visually
        output_filename = f"prediction_{name}.jpg"
        results[0].save(filename=output_filename)
        print(f"🖼️ Saved output to: {output_filename}")
        
    except Exception as e:
        print(f"❌ Status: Failed")
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if files exist
    if not os.path.exists(quantized_model):
        print(f"Error: {quantized_model} not found! Did you run quantize.py?")
    else:
        test_model(original_model, "Original")
        test_model(quantized_model, "Quantized")