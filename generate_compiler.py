import onnx
from onnx import shape_inference

def generate_compiler(onnx_path, output_cpp="yolo_auto_network.cpp"):
    print(f"🚀 Loading ONNX model for full compilation: {onnx_path}...")
    model = onnx.load(onnx_path)
    inferred_model = shape_inference.infer_shapes(model)

    # Dictionary to store tensor shapes: {tensor_name: (C, H, W)}
    tensor_shapes = {}
    for val in inferred_model.graph.value_info:
        dims = val.type.tensor_type.shape.dim
        if len(dims) == 4: # Batch, C, H, W
            tensor_shapes[val.name] = (dims[1].dim_value, dims[2].dim_value, dims[3].dim_value)

    # Initialize C++ code
    cpp_code = [
        "// --- AUTO-GENERATED YOLOv8 SCHEDULER ---",
        "void run_network(int8_t* cma_base_addr) {",
        "    size_t current_offset = 0;",
        "    auto alloc_buf = [&](size_t bytes) -> int8_t* {",
        "        int8_t* ptr = cma_base_addr + current_offset;",
        "        current_offset += bytes;",
        "        return ptr;",
        "    };",
        ""
    ]

    # Map ONNX tensor names to safe C++ variable names
    def sanitize(name):
        return "buf_" + name.replace("/", "_").replace(".", "_").replace("-", "_")

    # Track which tensors we've already allocated in C++
    allocated_tensors = set()

    node_counter = 0
    for node in inferred_model.graph.node:
        op = node.op_type
        inputs = node.input
        outputs = node.output
        
        # --- HANDLER: CONVOLUTION ---
        if op == "Conv":
            in_name = inputs[0]
            wt_name = inputs[1]
            out_name = outputs[0]
            
            if in_name not in tensor_shapes or out_name not in tensor_shapes: continue
            
            IC, H, W = tensor_shapes[in_name]
            OC, out_H, out_W = tensor_shapes[out_name]
            
            stride = 1
            for attr in node.attribute:
                if attr.name == "strides": stride = attr.ints[0]

            cpp_code.append(f"    // --- Node {node_counter}: Conv ---")
            
            # If the input buffer hasn't been declared yet (e.g., the very first image input)
            if in_name not in allocated_tensors:
                cpp_code.append(f"    int8_t* {sanitize(in_name)} = alloc_buf({IC * H * W});")
                allocated_tensors.add(in_name)

            cpp_code.append(f"    int8_t* {sanitize(out_name)} = alloc_buf({OC * out_H * out_W});")
            allocated_tensors.add(out_name)
            
            # Intermediate padded buffer
            cpp_code.append(f"    int8_t* pad_{node_counter} = alloc_buf({IC * (H+2) * (W+2)});")
            
            cpp_code.append(f"    cpu_pad({sanitize(in_name)}, pad_{node_counter}, {H}, {W}, {IC}, INPUT_ZP);")
            cpp_code.append(f"    execute_fpga_layer(pad_{node_counter}_phy, {sanitize(out_name)}_phy, wt_{node_counter}_phy, bias_{node_counter}_phy, {H+2}, {W+2}, {IC}, {OC}, 3, {stride}, SCALE_{node_counter});")
            cpp_code.append(f"    cpu_silu({sanitize(out_name)}, {OC * out_H * out_W}, SCALE_{node_counter}, ZP_{node_counter});\n")

        # --- HANDLER: RESIZE (UPSAMPLE) ---
        elif op == "Resize":
            in_name = inputs[0]
            out_name = outputs[0]
            if in_name not in tensor_shapes or out_name not in tensor_shapes: continue
            
            IC, H, W = tensor_shapes[in_name]
            _, out_H, out_W = tensor_shapes[out_name]
            scale = out_H // H

            cpp_code.append(f"    // --- Node {node_counter}: Upsample ---")
            cpp_code.append(f"    int8_t* {sanitize(out_name)} = alloc_buf({IC * out_H * out_W});")
            allocated_tensors.add(out_name)
            
            cpp_code.append(f"    cpu_upsample({sanitize(in_name)}, {sanitize(out_name)}, {H}, {W}, {IC}, {scale});\n")

        # --- HANDLER: CONCATENATION ---
        elif op == "Concat":
            out_name = outputs[0]
            if out_name not in tensor_shapes: continue
            
            OC, H, W = tensor_shapes[out_name]
            
            cpp_code.append(f"    // --- Node {node_counter}: Concat ---")
            cpp_code.append(f"    int8_t* {sanitize(out_name)} = alloc_buf({OC * H * W});")
            allocated_tensors.add(out_name)
            
            current_C_offset = 0
            for in_name in inputs:
                if in_name in tensor_shapes:
                    IC = tensor_shapes[in_name][0]
                    # Generate memory copy logic for the concat
                    cpp_code.append(f"    cpu_concat_append({sanitize(in_name)}, {IC}, {sanitize(out_name)}, {current_C_offset}, {H}, {W});")
                    current_C_offset += IC
            cpp_code.append("")

        # --- HANDLER: MAXPOOL (SPPF) ---
        elif op == "MaxPool":
            in_name = inputs[0]
            out_name = outputs[0]
            if in_name not in tensor_shapes or out_name not in tensor_shapes: continue
            
            C, H, W = tensor_shapes[in_name]
            
            cpp_code.append(f"    // --- Node {node_counter}: MaxPool ---")
            cpp_code.append(f"    int8_t* {sanitize(out_name)} = alloc_buf({C * H * W});")
            allocated_tensors.add(out_name)
            cpp_code.append(f"    cpu_maxpool_5x5({sanitize(in_name)}, {sanitize(out_name)}, {H}, {W}, {C});\n")
            
        node_counter += 1

    cpp_code.append("    std::cout << \"Total CMA Memory Used: \" << current_offset / (1024*1024) << \" MB\" << std::endl;")
    cpp_code.append("}\n")

    with open(output_cpp, "w") as f:
        f.write("\n".join(cpp_code))
    print(f"🎉 Fully routed C++ scheduler saved to: {output_cpp}")

if __name__ == "__main__":
    generate_compiler("yolov8n_final_int8_minmax.onnx")