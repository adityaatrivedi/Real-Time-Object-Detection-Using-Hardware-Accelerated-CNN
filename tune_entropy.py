import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import os
import cv2
import numpy as np
import glob

# --- CONFIG ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_MODEL = "yolov8n_entropy.onnx" # New name

# Find the dataset again
POSSIBLE_DIRS = [
    "./datasets/coco128/images/train2017",
    "../datasets/coco128/images/train2017",
    "/Users/adityatrivedi/Desktop/ARM-PROJECT/datasets/coco128/images/train2017",
    "datasets/coco128/images/train2017"
]

def find_dataset():
    for d in POSSIBLE_DIRS:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.jpg"))
            if len(files) > 0:
                print(f"✅ Found dataset at: {d} ({len(files)} images)")
                return files
    raise FileNotFoundError("Could not find COCO128 images.")

# --- PREPROCESSING ---
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

class COCOReader(CalibrationDataReader):
    def __init__(self, input_name, image_files):
        self.input_name = input_name
        self.enum = iter(image_files)
    
    def get_next(self):
        try:
            return {self.input_name: preprocess(next(self.enum))}
        except StopIteration:
            return None

def run_entropy_tuning():
    print("🚀 Starting Entropy (KL-Divergence) Quantization...")
    print("ℹ️  This will take longer than MinMax because it builds histograms.")
    
    image_files = find_dataset()
    model = onnx.load(INPUT_MODEL)
    dr = COCOReader(model.graph.input[0].name, image_files)
    
    quantize_static(
        INPUT_MODEL,
        OUTPUT_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        
        # --- THE FIX IS HERE ---
        calibrate_method=CalibrationMethod.Entropy,  # <--- CHANGED FROM MinMax
        # -----------------------
        
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        
        # Keep Safe Mode active
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'], 
        extra_options={'ActivationSymmetric': False}
    )
    print(f"🎉 Entropy Model Saved: {OUTPUT_MODEL}")

if __name__ == "__main__":
    run_entropy_tuning()