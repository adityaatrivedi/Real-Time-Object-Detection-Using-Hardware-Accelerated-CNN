import onnx
from onnx import numpy_helper
import numpy as np
import os

# --- CONFIG ---
MODEL_PATH = "yolov8n_final_int8_minmax.onnx" # Your quantized model
OUTPUT_DIR = "binary_weights_minmax"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_onnx_weights():
    print(f"Loading {MODEL_PATH}...")
    model = onnx.load(MODEL_PATH)
    
    print(f"\n--- Extracting INT8 Weights to {OUTPUT_DIR}/ ---")
    
    with open(f"{OUTPUT_DIR}/memory_map.txt", "w") as map_file:
        map_file.write("Tensor_Name, Filename, Size_Bytes, Data_Type\n")
        
        # In ONNX, weights are stored in the "Initializer" list
        for tensor in model.graph.initializer:
            # Convert ONNX tensor to Numpy
            data = numpy_helper.to_array(tensor)
            
            # Flatten
            data_flat = data.flatten()
            
            # Clean filename
            clean_name = tensor.name.replace("/", "_").replace(".", "_")
            filename = f"{clean_name}.bin"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save Raw Binary
            data_flat.tofile(filepath)
            
            # Log
            size_bytes = data_flat.nbytes
            dtype = str(data.dtype)
            
            # Filter: Only save things that look like weights (not tiny scalars)
            if size_bytes > 128: 
                print(f"Saved: {filename} | {size_bytes} bytes | {dtype}")
                map_file.write(f"{tensor.name}, {filename}, {size_bytes}, {dtype}\n")

    print(f"\n✅ Done! Use 'memory_map.txt' to calculate your 0x8000_0000 offsets.")

if __name__ == "__main__":
    extract_onnx_weights()