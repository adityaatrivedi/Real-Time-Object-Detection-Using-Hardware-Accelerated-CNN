import onnx
from onnx import numpy_helper
import numpy as np
import os
import struct

# --- CONFIGURATION ---
MODEL_PATH = "yolov8n_final_int8.onnx"  # Your final quantized model
OUTPUT_DIR = "fpga_params"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_all():
    print(f"Loading {MODEL_PATH}...")
    model = onnx.load(MODEL_PATH)
    graph = model.graph
    
    # Create a mapping of {Output_Name: Node} to easily trace back inputs
    node_map = {node.output[0]: node for node in graph.node}
    
    # Create a mapping of {Initializer_Name: Numpy_Array} for weights/biases
    init_map = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    print(f"\n--- Extracting to {OUTPUT_DIR}/ ---")
    
    # We create a manifest file for your C code to read
    manifest_path = os.path.join(OUTPUT_DIR, "layers_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("Layer_ID, Name, Weights_File, Bias_File, Scale_File, Out_Ch, In_Ch\n")

        layer_id = 0
        for node in graph.node:
            if node.op_type == "Conv":
                print(f"Processing Layer {layer_id}: {node.name}")
                
                # --- 1. GET WEIGHTS (INT8) ---
                # In a QDQ model, Conv input[1] is the output of a DequantizeLinear node
                weight_dq_name = node.input[1] 
                
                if weight_dq_name not in node_map:
                    print(f"  ⚠️ Skipping {node.name} (Weights not quantized?)")
                    continue
                    
                dq_node = node_map[weight_dq_name] # This is the DequantizeLinear node
                
                # The inputs to DQ are [Int8_Weights, Scale, Zero_Point]
                raw_weight_name = dq_node.input[0]
                scale_name = dq_node.input[1]
                
                if raw_weight_name not in init_map:
                    print(f"  ⚠️ Could not find initializer for {raw_weight_name}")
                    continue
                
                # Extract INT8 Weights
                weights = init_map[raw_weight_name]
                w_filename = f"layer{layer_id}_weights.bin"
                weights.flatten().tofile(os.path.join(OUTPUT_DIR, w_filename))
                
                # --- 2. GET SCALES (FLOAT32) ---
                # The scale is usually a scalar or a 1D tensor (per-channel)
                if scale_name in init_map:
                    scale_val = init_map[scale_name]
                    # FPGA usually wants one scalar per layer (Per-Tensor)
                    # If it's Per-Channel, you might need to save all floats.
                    s_filename = f"layer{layer_id}_scale.bin"
                    # Force to float32 just in case
                    scale_val.astype(np.float32).tofile(os.path.join(OUTPUT_DIR, s_filename))
                else:
                    s_filename = "NONE"

                # --- 3. GET BIAS (INT32 or FLOAT32) ---
                # Conv input[2] is bias (optional)
                b_filename = "NONE"
                if len(node.input) > 2:
                    bias_name = node.input[2]
                    # Sometimes bias comes from a DQ node too, sometimes it's raw
                    if bias_name in node_map and node_map[bias_name].op_type == "DequantizeLinear":
                         # Trace back one more step if bias is quantized
                         bias_dq = node_map[bias_name]
                         bias_raw_name = bias_dq.input[0]
                         if bias_raw_name in init_map:
                             bias_data = init_map[bias_raw_name]
                             b_filename = f"layer{layer_id}_bias.bin"
                             bias_data.tofile(os.path.join(OUTPUT_DIR, b_filename))
                    elif bias_name in init_map:
                        # Raw bias (Float32)
                        bias_data = init_map[bias_name]
                        b_filename = f"layer{layer_id}_bias.bin"
                        bias_data.tofile(os.path.join(OUTPUT_DIR, b_filename))

                # --- 4. LOGGING ---
                out_ch = weights.shape[0]
                in_ch = weights.shape[1]
                
                # Write to manifest
                f.write(f"{layer_id}, {node.name}, {w_filename}, {b_filename}, {s_filename}, {out_ch}, {in_ch}\n")
                
                print(f"  -> Saved: W={w_filename}, B={b_filename}, S={s_filename} (Shape: {out_ch}x{in_ch})")
                layer_id += 1

    print(f"\n✅ Done! Files saved to {OUTPUT_DIR}/")
    print(f"   Manifest: {manifest_path}")

if __name__ == "__main__":
    extract_all()