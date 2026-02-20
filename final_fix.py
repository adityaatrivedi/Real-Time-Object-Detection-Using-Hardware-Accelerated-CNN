import os
import shutil
import requests
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from ultralytics import YOLO

# --- CONFIG ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_MODEL = "yolov8n_quantized.onnx"
CALIB_DIR = "calibration_images_fixed"

# --- 1. PREPARE DATA (Same as before, just ensuring we have it) ---
def get_data():
    if not os.path.exists(CALIB_DIR):
        os.makedirs(CALIB_DIR)
        urls = ["https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
                "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg"]
        print("Downloading images...")
        for i, url in enumerate(urls):
            with open(f"{CALIB_DIR}/img_{i}.jpg", "wb") as f:
                f.write(requests.get(url).content)

# --- 2. PREPROCESS (Strict YOLOv8 formatting) ---
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # Letterbox resize (Fit 640x640 with gray padding)
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = (640, 640)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Add gray border (114)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # HWC to CHW, BGR to RGB, Normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, axis=0)
    return img

class YoloReader(CalibrationDataReader):
    def __init__(self, input_name):
        self.input_name = input_name
        self.files = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(".jpg")]
        self.enum = iter(self.files)

    def get_next(self):
        try:
            return {self.input_name: preprocess(next(self.enum))}
        except StopIteration:
            return None

# --- 3. QUANTIZE (The Critical Change) ---
def quantize():
    print(f"Quantizing {INPUT_MODEL}...")
    model = onnx.load(INPUT_MODEL)
    input_name = model.graph.input[0].name
    
    dr = YoloReader(input_name)
    
    # CRITICAL FIX: 
    # activation_type = QUInt8 (Unsigned 8-bit)
    # weight_type = QInt8 (Signed 8-bit)
    # This combination prevents the "Silencing" issue in YOLO detection heads.
    
    quantize_static(
        INPUT_MODEL,
        OUTPUT_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, 
        calibrate_method=CalibrationMethod.MinMax, # MinMax is safer for detection
        weight_type=QuantType.QInt8,    
        activation_type=QuantType.QUInt8, # <--- THIS IS THE FIX
        extra_options={'ActivationSymmetric': False} # Essential for QUInt8
    )
    print("Quantization Complete.")

# --- 4. VERIFY ---
def verify():
    print("Verifying...")
    try:
        model = YOLO(OUTPUT_MODEL, task='detect')
        # Force the model to accept the quantized input structure
        results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            print(f"✅ SUCCESS! Detected {len(boxes)} objects.")
            results[0].save(filename="final_success.jpg")
        else:
            print("❌ Still 0 detections. The output head might need exclusion.")
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    get_data()
    quantize()
    verify()