import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import os
import cv2
import numpy as np
import glob

# --- CONFIG ---
# INPUT: The file you just created
INPUT_MODEL = "yolov8n_final_pruned.onnx" 
# OUTPUT: The final file for the FPGA
OUTPUT_MODEL = "yolov8n_final_int8.onnx"

# --- STANDARD SETUP (Same as before) ---
def find_dataset():
    POSSIBLE_DIRS = ["./datasets/coco128/images/train2017", "datasets/coco128/images/train2017"]
    for d in POSSIBLE_DIRS:
        if os.path.exists(d): return glob.glob(os.path.join(d, "*.jpg"))
    raise FileNotFoundError("Could not find COCO128 images.")

def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    h, w = img.shape[:2]
    r = min(640/h, 640/w)
    new_unpad = int(round(w*r)), int(round(h*r))
    dw, dh = (640 - new_unpad[0])/2, (640 - new_unpad[1])/2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

class COCOReader(CalibrationDataReader):
    def __init__(self, input_name, image_files):
        self.input_name = input_name
        self.enum = iter(image_files)
    def get_next(self):
        try: return {self.input_name: preprocess(next(self.enum))}
        except StopIteration: return None

def run_tuning():
    print(f"🚀 Quantizing: {INPUT_MODEL} ...")
    image_files = find_dataset()
    model = onnx.load(INPUT_MODEL)
    dr = COCOReader(model.graph.input[0].name, image_files)
    
    quantize_static(
        INPUT_MODEL, OUTPUT_MODEL, calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.Entropy,
        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        op_types_to_quantize=['Conv', 'MatMul', 'Gemm'], 
        extra_options={'ActivationSymmetric': False}
    )
    print(f"🎉 DONE! Final FPGA Model: {OUTPUT_MODEL}")

if __name__ == "__main__":
    run_tuning()