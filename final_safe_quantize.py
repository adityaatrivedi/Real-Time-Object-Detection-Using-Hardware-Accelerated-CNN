import os
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from ultralytics import YOLO
import cv2
import numpy as np

# --- CONFIG ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_MODEL = "yolov8n_safe_hybrid.onnx"
CALIB_DIR = "calibration_images_fixed"

# --- 1. DATA READER (Standard) ---
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    r = min(640/h, 640/w)
    new_unpad = int(round(w*r)), int(round(h*r))
    dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
    dw, dh = dw/2, dh/2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

class YoloReader(CalibrationDataReader):
    def __init__(self, input_name):
        self.input_name = input_name
        self.files = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith(".jpg")]
        self.enum = iter(self.files)
    def get_next(self):
        try: return {self.input_name: preprocess(next(self.enum))}
        except StopIteration: return None

# --- 2. QUANTIZE (The Safe Way) ---
def run_safe_quantization():
    print(f"⚙️  Quantizing ONLY Convolutions (Safe Mode)...")
    model = onnx.load(INPUT_MODEL)
    dr = YoloReader(model.graph.input[0].name)
    
    quantize_static(
        INPUT_MODEL,
        OUTPUT_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        
        # --- THE FIX IS HERE ---
        # We explicitly list ONLY the layers that are safe to quantize.
        # This forces 'Add', 'Sub', 'Mul', 'Div', 'Concat' to stay as Float32.
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'], 
        
        extra_options={'ActivationSymmetric': False}
    )
    print(f"🎉 Success: {OUTPUT_MODEL}")

# --- 3. VERIFY ---
def verify():
    print("\n--- Testing Safe Hybrid Model ---")
    try:
        model = YOLO(OUTPUT_MODEL, task='detect')
        results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            print(f"✅ SUCCESS! Detected {len(boxes)} objects.")
            results[0].save(filename="safe_result.jpg")
            print("🖼️ Saved result to safe_result.jpg")
        else:
            print("❌ Failed: Still 0 detections.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_safe_quantization()
    verify()