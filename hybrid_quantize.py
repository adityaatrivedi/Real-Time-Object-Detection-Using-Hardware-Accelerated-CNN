import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import os
import cv2
from ultralytics import YOLO

# --- CONFIG ---
INPUT_MODEL = "yolov8n.onnx"
OUTPUT_MODEL = "yolov8n_hybrid.onnx"
CALIB_DIR = "calibration_images_fixed"

# --- 1. SMART HEAD DETECTION ---
def get_sensitive_nodes(model_path):
    """
    Walks backwards from the model output to find the 'Head' nodes 
    (Concat, Reshape, Mul, Sigmoid) that must stay Float32.
    """
    model = onnx.load(model_path)
    exclude_nodes = []
    
    # Get the names of the final output tensors
    output_names = [output.name for output in model.graph.output]
    
    # Iterate through all nodes to find the ones producing these outputs
    print(f"🔍 Tracing back from outputs: {output_names}")
    
    # We want to exclude the last 2-3 layers of operations
    # This is a heuristic that works for most YOLOv8 exports
    for node in model.graph.node:
        # If a node's output connects to the graph's final output
        if any(out in output_names for out in node.output):
            exclude_nodes.append(node.name)
            # Find inputs to this node to exclude them too (one level deeper)
            for inp in node.input:
                parent_nodes = [n for n in model.graph.node if inp in n.output]
                for p_node in parent_nodes:
                    exclude_nodes.append(p_node.name)
    
    # Remove duplicates
    exclude_nodes = list(set(exclude_nodes))
    print(f"🛑 Excluding {len(exclude_nodes)} 'Head' nodes from quantization to preserve accuracy.")
    return exclude_nodes

# --- 2. DATA READER (Same as before) ---
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    # Letterbox resize logic (Simplified for brevity, assumes previous script ran setup)
    h, w = img.shape[:2]
    r = min(640/h, 640/w)
    new_unpad = int(round(w*r)), int(round(h*r))
    img = cv2.resize(img, new_unpad)
    # Pad to 640x640
    dw, dh = 640 - new_unpad[0], 640 - new_unpad[1]
    img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(114,114,114))
    
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

# --- 3. QUANTIZE WITH EXCLUSIONS ---
def run_hybrid_quantization():
    # 1. Identify Nodes to Exclude
    exclusions = get_sensitive_nodes(INPUT_MODEL)
    
    # 2. Setup Reader
    model = onnx.load(INPUT_MODEL)
    dr = YoloReader(model.graph.input[0].name)
    
    # 3. Quantize
    print(f"⚙️  Quantizing body to INT8, keeping head in FP32...")
    quantize_static(
        INPUT_MODEL,
        OUTPUT_MODEL,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        nodes_to_exclude=exclusions,  # <--- MAGIC FIX
        extra_options={'ActivationSymmetric': False}
    )
    print(f"🎉 Success: {OUTPUT_MODEL}")

# --- 4. VERIFY ---
def verify():
    print("\n--- Testing Hybrid Model ---")
    model = YOLO(OUTPUT_MODEL, task='detect')
    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
    print(f"📦 Objects Detected: {len(results[0].boxes)}")
    if len(results[0].boxes) > 0:
        results[0].save(filename="hybrid_success.jpg")
        print("✅ Saved to hybrid_success.jpg")

if __name__ == "__main__":
    run_hybrid_quantization()
    verify()