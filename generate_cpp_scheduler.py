import onnx
from onnx import shape_inference

def generate_scheduler(onnx_path, output_cpp="yolo_scheduler.cpp"):
    print(f"Loading ONNX model: {onnx_path}...")
    model = onnx.load(onnx_path)
    
    # Infer shapes so we know the Height and Width at every layer
    try:
        inferred_model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Shape inference warning (safe to ignore if shapes exist): {e}")
        inferred_model = model

    # Create a dictionary of all tensor shapes in the graph
    value_info = {val.name: val for val in inferred_model.graph.value_info}
    for val in inferred_model.graph.input:
        value_info[val.name] = val
    for val in inferred_model.graph.output:
        value_info[val.name] = val

    def get_shape(tensor_name):
        if tensor_name in value_info:
            dims = value_info[tensor_name].type.tensor_type.shape.dim
            return [d.dim_value for d in dims]
        return None

    cpp_code = ["void run_network() {", "    // --- AUTO-GENERATED FPGA SCHEDULER ---\n"]
    layer_counter = 0

    for node in inferred_model.graph.node:
        if node.op_type == "Conv":
            input_name = node.input[0]
            weight_name = node.input[1]
            output_name = node.output[0]

            # Get shapes
            in_shape = get_shape(input_name)
            wt_shape = get_shape(weight_name)
            
            if not in_shape or not wt_shape:
                continue

            # Parse dimensions
            _, IC, H, W = in_shape
            OC, _, K, _ = wt_shape
            
            # Parse attributes (stride, padding)
            stride = 1
            for attr in node.attribute:
                if attr.name == "strides":
                    stride = attr.ints[0]

            # Write C++ Code for this layer
            cpp_code.append(f"    // --- Layer {layer_counter} ({node.name}) ---")
            cpp_code.append(f"    // Input: {H}x{W}x{IC} | Output Channels: {OC} | Kernel: {K}x{K} | Stride: {stride}")
            
            # 1. Padding Call
            cpp_code.append(f"    cpu_pad(buf_{layer_counter}_in, buf_{layer_counter}_pad, {H}, {W}, {IC}, INPUT_ZP);")
            
            # 2. FPGA Call
            # We add +2 to H and W because the CPU padded it to simulate SAME padding
            cpp_code.append(f"    execute_fpga_layer(buf_{layer_counter}_pad_phy, buf_{layer_counter}_out_phy, wt_{layer_counter}_phy, bias_{layer_counter}_phy, {H+2}, {W+2}, {IC}, {OC}, {K}, {stride}, LAYER_{layer_counter}_SCALE);")
            
            # 3. SiLU Activation Call (assuming next node isn't the final output)
            new_H = H // stride
            new_W = W // stride
            cpp_code.append(f"    cpu_silu(buf_{layer_counter}_out_virt, {new_H} * {new_W} * {OC}, LAYER_{layer_counter}_SCALE, LAYER_{layer_counter}_ZP);\n")

            layer_counter += 1

    cpp_code.append("}\n")

    # Save to file
    with open(output_cpp, "w") as f:
        f.write("\n".join(cpp_code))
    
    print(f"🎉 Successfully generated C++ scheduler for {layer_counter} Convolutional layers!")
    print(f"Saved to: {output_cpp}")

if __name__ == "__main__":
    # Point this to your INT8 MinMax ONNX file
    generate_scheduler("yolov8n_final_int8_minmax.onnx")