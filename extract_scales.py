import onnx
from onnx import numpy_helper

def extract_scales(onnx_path):
    print(f"Loading ONNX model: {onnx_path}...")
    model = onnx.load(onnx_path)
    
    # Create a dictionary of all initializers (where scales are stored)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    
    cpp_code = ["    // --- AUTO-EXTRACTED SCALES AND ZERO POINTS ---"]
    
    node_counter = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            # In a quantized ONNX, the input to Conv usually comes from a QuantizeLinear node,
            # or the weights are quantized. The output goes to a DequantizeLinear or QuantizeLinear.
            
            # To get the exact scale for this layer's output, we look for the node 
            # that consumes this Conv's output.
            conv_output_name = node.output[0]
            
            layer_scale = 0.005 # Default fallback
            layer_zp = 0
            
            # Search the graph for the Dequantize/Quantize node connected to this Conv's output
            for next_node in model.graph.node:
                if next_node.op_type in ["QuantizeLinear", "DequantizeLinear"] and conv_output_name in next_node.input:
                    scale_name = next_node.input[1]
                    zp_name = next_node.input[2] if len(next_node.input) > 2 else None
                    
                    if scale_name in initializers:
                        # Scales are usually 1D arrays with 1 element
                        val = initializers[scale_name]
                        layer_scale = val.item() if val.size == 1 else val[0]
                    
                    if zp_name and zp_name in initializers:
                        val = initializers[zp_name]
                        layer_zp = int(val.item() if val.size == 1 else val[0])
                    break
            
            # Write the C++ line
            cpp_code.append(f"    layer_scales[{node_counter}] = {layer_scale}f;")
            cpp_code.append(f"    layer_zps[{node_counter}] = {layer_zp};")
            
            node_counter += 1

    # Save to a text file
    with open("c++_scales.txt", "w") as f:
        f.write("\n".join(cpp_code))
    
    print("Done! Open c++_scales.txt and paste it into your C++ app.")

if __name__ == "__main__":
    extract_scales("yolov8n_final_int8_minmax.onnx")