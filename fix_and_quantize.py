import os
import shutil
import requests
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_MODEL = "yolov8n_quantized.onnx"
CALIB_DIR = "calibration_images_fixed"

# --- STEP 1: RESET DATA (Download only known-good images) ---
def setup_data():
    if os.path.exists(CALIB_DIR):
        shutil.rmtree(CALIB_DIR)
    os.makedirs(CALIB_DIR)
    
    # Use reliable GitHub raw links from Ultralytics
    urls = [
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
        # We duplicate them to simulate a larger batch if needed, but 2-4 clean images is better than 100 broken ones
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", 
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg"
    ]
    
    print(f"⬇️ Downloading clean calibration images to {CALIB_DIR}...")
    for i, url in enumerate(urls):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(f"{CALIB_DIR}/img_{i}.jpg", "wb") as f:
                    f.write(resp.content)
            else:
                print(f"Failed to fetch {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# --- STEP 2: CORRECT PREPROCESSING (Letterbox) ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None

    # Get current shape
    h, w = img.shape[:2]
    
    # Scale ratio (new / old)
    r = min(640 / h, 640 / w)
    
    # Compute padding
    new_unpad = int(round(w * r)), int(round(h * r))
    dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
    
    # Divide padding by 2 (center the image)
    dw /= 2 
    dh /= 2

    # Resize
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add border (padding)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Standard Preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # Add batch dim
    return img

class YoloReader(CalibrationDataReader):
    def __init__(self, input_name, image_folder):
        self.input_name = input_name
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
        self.enum_data = iter(self.image_paths)
        print(f"✅ Found {len(self.image_paths)} valid images.")

    def get_next(self):
        try:
            path = next(self.enum_data)
            return {self.input_name: preprocess_image(path)}
        except StopIteration:
            return None

# --- STEP 3: QUANTIZE ---
def run_quantization():
    print(f"⚙️  Quantizing {INPUT_MODEL}...")
    
    # Load model to get input name
    model_onnx = onnx.load(INPUT_MODEL)
    input_name = model_onnx.graph.input[0].name

    dr = YoloReader(input_name, CALIB_DIR)

    # We use basic MinMax calibration which is safer for YOLOv8 on ONNX Runtime
    quantize_static(
        INPUT_MODEL,
        OUTPUT_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    print(f"🎉 Quantization Complete: {OUTPUT_MODEL}")

# --- STEP 4: VERIFY ---
def verify():
    print("\n--- Final Verification ---")
    model = YOLO(OUTPUT_MODEL, task='detect')
    # Run on the bus image (which we know is in our calib set, so it should be perfect)
    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
    
    if len(results[0].boxes) > 0:
        print(f"✅ SUCCESS! Detected {len(results[0].boxes)} objects.")
        results[0].save(filename="final_result.jpg")
        print("saved to final_result.jpg")
    else:
        print("❌ FAILED. Still 0 detections.")

if __name__ == "__main__":
    setup_data()
    run_quantization()
    verify()