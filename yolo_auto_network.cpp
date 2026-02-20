// --- AUTO-GENERATED YOLOv8 SCHEDULER ---
void run_network(int8_t* cma_base_addr) {
    size_t current_offset = 0;
    auto alloc_buf = [&](size_t bytes) -> int8_t* {
        int8_t* ptr = cma_base_addr + current_offset;
        current_offset += bytes;
        return ptr;
    };

    // --- Node 129: Conv ---
    int8_t* buf_images_DequantizeLinear_Output = alloc_buf(1228800);
    int8_t* buf__model_0_conv_Conv_output_0 = alloc_buf(1638400);
    int8_t* pad_129 = alloc_buf(1236492);
    cpu_pad(buf_images_DequantizeLinear_Output, pad_129, 640, 640, 3, INPUT_ZP);
    execute_fpga_layer(pad_129_phy, buf__model_0_conv_Conv_output_0_phy, wt_129_phy, bias_129_phy, 642, 642, 3, 16, 3, 2, SCALE_129);
    cpu_silu(buf__model_0_conv_Conv_output_0, 1638400, SCALE_129, ZP_129);

    // --- Node 136: Conv ---
    int8_t* buf__model_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(1638400);
    int8_t* buf__model_1_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_136 = alloc_buf(1658944);
    cpu_pad(buf__model_0_act_Mul_output_0_DequantizeLinear_Output, pad_136, 320, 320, 16, INPUT_ZP);
    execute_fpga_layer(pad_136_phy, buf__model_1_conv_Conv_output_0_phy, wt_136_phy, bias_136_phy, 322, 322, 16, 32, 3, 2, SCALE_136);
    cpu_silu(buf__model_1_conv_Conv_output_0, 819200, SCALE_136, ZP_136);

    // --- Node 143: Conv ---
    int8_t* buf__model_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(819200);
    int8_t* buf__model_2_cv1_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_143 = alloc_buf(839808);
    cpu_pad(buf__model_1_act_Mul_output_0_DequantizeLinear_Output, pad_143, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(pad_143_phy, buf__model_2_cv1_conv_Conv_output_0_phy, wt_143_phy, bias_143_phy, 162, 162, 32, 32, 3, 1, SCALE_143);
    cpu_silu(buf__model_2_cv1_conv_Conv_output_0, 819200, SCALE_143, ZP_143);

    // --- Node 151: Conv ---
    int8_t* buf__model_2_Split_output_1_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_2_m_0_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_151 = alloc_buf(419904);
    cpu_pad(buf__model_2_Split_output_1_DequantizeLinear_Output, pad_151, 160, 160, 16, INPUT_ZP);
    execute_fpga_layer(pad_151_phy, buf__model_2_m_0_cv1_conv_Conv_output_0_phy, wt_151_phy, bias_151_phy, 162, 162, 16, 8, 3, 1, SCALE_151);
    cpu_silu(buf__model_2_m_0_cv1_conv_Conv_output_0, 204800, SCALE_151, ZP_151);

    // --- Node 158: Conv ---
    int8_t* buf__model_2_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_2_m_0_cv2_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_158 = alloc_buf(209952);
    cpu_pad(buf__model_2_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_158, 160, 160, 8, INPUT_ZP);
    execute_fpga_layer(pad_158_phy, buf__model_2_m_0_cv2_conv_Conv_output_0_phy, wt_158_phy, bias_158_phy, 162, 162, 8, 16, 3, 1, SCALE_158);
    cpu_silu(buf__model_2_m_0_cv2_conv_Conv_output_0, 409600, SCALE_158, ZP_158);

    // --- Node 164: Concat ---
    int8_t* buf__model_2_Concat_output_0 = alloc_buf(1228800);
    cpu_concat_append(buf__model_2_Split_output_0, 16, buf__model_2_Concat_output_0, 0, 160, 160);
    cpu_concat_append(buf__model_2_Split_output_1_DequantizeLinear_Output, 16, buf__model_2_Concat_output_0, 16, 160, 160);
    cpu_concat_append(buf__model_2_m_0_Add_output_0, 16, buf__model_2_Concat_output_0, 32, 160, 160);

    // --- Node 167: Conv ---
    int8_t* buf__model_2_Concat_output_0_DequantizeLinear_Output = alloc_buf(1228800);
    int8_t* buf__model_2_cv2_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_167 = alloc_buf(1259712);
    cpu_pad(buf__model_2_Concat_output_0_DequantizeLinear_Output, pad_167, 160, 160, 48, INPUT_ZP);
    execute_fpga_layer(pad_167_phy, buf__model_2_cv2_conv_Conv_output_0_phy, wt_167_phy, bias_167_phy, 162, 162, 48, 32, 3, 1, SCALE_167);
    cpu_silu(buf__model_2_cv2_conv_Conv_output_0, 819200, SCALE_167, ZP_167);

    // --- Node 174: Conv ---
    int8_t* buf__model_2_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(819200);
    int8_t* buf__model_3_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_174 = alloc_buf(839808);
    cpu_pad(buf__model_2_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_174, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(pad_174_phy, buf__model_3_conv_Conv_output_0_phy, wt_174_phy, bias_174_phy, 162, 162, 32, 64, 3, 2, SCALE_174);
    cpu_silu(buf__model_3_conv_Conv_output_0, 409600, SCALE_174, ZP_174);

    // --- Node 181: Conv ---
    int8_t* buf__model_3_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_4_cv1_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_181 = alloc_buf(430336);
    cpu_pad(buf__model_3_act_Mul_output_0_DequantizeLinear_Output, pad_181, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_181_phy, buf__model_4_cv1_conv_Conv_output_0_phy, wt_181_phy, bias_181_phy, 82, 82, 64, 64, 3, 1, SCALE_181);
    cpu_silu(buf__model_4_cv1_conv_Conv_output_0, 409600, SCALE_181, ZP_181);

    // --- Node 189: Conv ---
    int8_t* buf__model_4_Split_output_1_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_4_m_0_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_189 = alloc_buf(215168);
    cpu_pad(buf__model_4_Split_output_1_DequantizeLinear_Output, pad_189, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_189_phy, buf__model_4_m_0_cv1_conv_Conv_output_0_phy, wt_189_phy, bias_189_phy, 82, 82, 32, 16, 3, 1, SCALE_189);
    cpu_silu(buf__model_4_m_0_cv1_conv_Conv_output_0, 102400, SCALE_189, ZP_189);

    // --- Node 196: Conv ---
    int8_t* buf__model_4_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_4_m_0_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_196 = alloc_buf(107584);
    cpu_pad(buf__model_4_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_196, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_196_phy, buf__model_4_m_0_cv2_conv_Conv_output_0_phy, wt_196_phy, bias_196_phy, 82, 82, 16, 32, 3, 1, SCALE_196);
    cpu_silu(buf__model_4_m_0_cv2_conv_Conv_output_0, 204800, SCALE_196, ZP_196);

    // --- Node 204: Conv ---
    int8_t* buf__model_4_m_0_Add_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_4_m_1_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_204 = alloc_buf(215168);
    cpu_pad(buf__model_4_m_0_Add_output_0_DequantizeLinear_Output, pad_204, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_204_phy, buf__model_4_m_1_cv1_conv_Conv_output_0_phy, wt_204_phy, bias_204_phy, 82, 82, 32, 16, 3, 1, SCALE_204);
    cpu_silu(buf__model_4_m_1_cv1_conv_Conv_output_0, 102400, SCALE_204, ZP_204);

    // --- Node 211: Conv ---
    int8_t* buf__model_4_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_4_m_1_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_211 = alloc_buf(107584);
    cpu_pad(buf__model_4_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_211, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_211_phy, buf__model_4_m_1_cv2_conv_Conv_output_0_phy, wt_211_phy, bias_211_phy, 82, 82, 16, 32, 3, 1, SCALE_211);
    cpu_silu(buf__model_4_m_1_cv2_conv_Conv_output_0, 204800, SCALE_211, ZP_211);

    // --- Node 217: Concat ---
    int8_t* buf__model_4_Concat_output_0 = alloc_buf(819200);
    cpu_concat_append(buf__model_4_Split_output_0, 32, buf__model_4_Concat_output_0, 0, 80, 80);
    cpu_concat_append(buf__model_4_Split_output_1_DequantizeLinear_Output, 32, buf__model_4_Concat_output_0, 32, 80, 80);
    cpu_concat_append(buf__model_4_m_0_Add_output_0_DequantizeLinear_Output, 32, buf__model_4_Concat_output_0, 64, 80, 80);
    cpu_concat_append(buf__model_4_m_1_Add_output_0, 32, buf__model_4_Concat_output_0, 96, 80, 80);

    // --- Node 220: Conv ---
    int8_t* buf__model_4_Concat_output_0_DequantizeLinear_Output = alloc_buf(819200);
    int8_t* buf__model_4_cv2_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_220 = alloc_buf(860672);
    cpu_pad(buf__model_4_Concat_output_0_DequantizeLinear_Output, pad_220, 80, 80, 128, INPUT_ZP);
    execute_fpga_layer(pad_220_phy, buf__model_4_cv2_conv_Conv_output_0_phy, wt_220_phy, bias_220_phy, 82, 82, 128, 64, 3, 1, SCALE_220);
    cpu_silu(buf__model_4_cv2_conv_Conv_output_0, 409600, SCALE_220, ZP_220);

    // --- Node 227: Conv ---
    int8_t* buf__model_4_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_5_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_227 = alloc_buf(430336);
    cpu_pad(buf__model_4_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_227, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_227_phy, buf__model_5_conv_Conv_output_0_phy, wt_227_phy, bias_227_phy, 82, 82, 64, 128, 3, 2, SCALE_227);
    cpu_silu(buf__model_5_conv_Conv_output_0, 204800, SCALE_227, ZP_227);

    // --- Node 234: Conv ---
    int8_t* buf__model_5_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_6_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_234 = alloc_buf(225792);
    cpu_pad(buf__model_5_act_Mul_output_0_DequantizeLinear_Output, pad_234, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_234_phy, buf__model_6_cv1_conv_Conv_output_0_phy, wt_234_phy, bias_234_phy, 42, 42, 128, 128, 3, 1, SCALE_234);
    cpu_silu(buf__model_6_cv1_conv_Conv_output_0, 204800, SCALE_234, ZP_234);

    // --- Node 242: Conv ---
    int8_t* buf__model_6_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_6_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_242 = alloc_buf(112896);
    cpu_pad(buf__model_6_Split_output_1_DequantizeLinear_Output, pad_242, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_242_phy, buf__model_6_m_0_cv1_conv_Conv_output_0_phy, wt_242_phy, bias_242_phy, 42, 42, 64, 32, 3, 1, SCALE_242);
    cpu_silu(buf__model_6_m_0_cv1_conv_Conv_output_0, 51200, SCALE_242, ZP_242);

    // --- Node 249: Conv ---
    int8_t* buf__model_6_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_6_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_249 = alloc_buf(56448);
    cpu_pad(buf__model_6_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_249, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_249_phy, buf__model_6_m_0_cv2_conv_Conv_output_0_phy, wt_249_phy, bias_249_phy, 42, 42, 32, 64, 3, 1, SCALE_249);
    cpu_silu(buf__model_6_m_0_cv2_conv_Conv_output_0, 102400, SCALE_249, ZP_249);

    // --- Node 257: Conv ---
    int8_t* buf__model_6_m_0_Add_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_6_m_1_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_257 = alloc_buf(112896);
    cpu_pad(buf__model_6_m_0_Add_output_0_DequantizeLinear_Output, pad_257, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_257_phy, buf__model_6_m_1_cv1_conv_Conv_output_0_phy, wt_257_phy, bias_257_phy, 42, 42, 64, 32, 3, 1, SCALE_257);
    cpu_silu(buf__model_6_m_1_cv1_conv_Conv_output_0, 51200, SCALE_257, ZP_257);

    // --- Node 264: Conv ---
    int8_t* buf__model_6_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_6_m_1_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_264 = alloc_buf(56448);
    cpu_pad(buf__model_6_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_264, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_264_phy, buf__model_6_m_1_cv2_conv_Conv_output_0_phy, wt_264_phy, bias_264_phy, 42, 42, 32, 64, 3, 1, SCALE_264);
    cpu_silu(buf__model_6_m_1_cv2_conv_Conv_output_0, 102400, SCALE_264, ZP_264);

    // --- Node 270: Concat ---
    int8_t* buf__model_6_Concat_output_0 = alloc_buf(409600);
    cpu_concat_append(buf__model_6_Split_output_0, 64, buf__model_6_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_6_Split_output_1_DequantizeLinear_Output, 64, buf__model_6_Concat_output_0, 64, 40, 40);
    cpu_concat_append(buf__model_6_m_0_Add_output_0_DequantizeLinear_Output, 64, buf__model_6_Concat_output_0, 128, 40, 40);
    cpu_concat_append(buf__model_6_m_1_Add_output_0, 64, buf__model_6_Concat_output_0, 192, 40, 40);

    // --- Node 273: Conv ---
    int8_t* buf__model_6_Concat_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_6_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_273 = alloc_buf(451584);
    cpu_pad(buf__model_6_Concat_output_0_DequantizeLinear_Output, pad_273, 40, 40, 256, INPUT_ZP);
    execute_fpga_layer(pad_273_phy, buf__model_6_cv2_conv_Conv_output_0_phy, wt_273_phy, bias_273_phy, 42, 42, 256, 128, 3, 1, SCALE_273);
    cpu_silu(buf__model_6_cv2_conv_Conv_output_0, 204800, SCALE_273, ZP_273);

    // --- Node 280: Conv ---
    int8_t* buf__model_6_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_7_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_280 = alloc_buf(225792);
    cpu_pad(buf__model_6_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_280, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_280_phy, buf__model_7_conv_Conv_output_0_phy, wt_280_phy, bias_280_phy, 42, 42, 128, 256, 3, 2, SCALE_280);
    cpu_silu(buf__model_7_conv_Conv_output_0, 102400, SCALE_280, ZP_280);

    // --- Node 287: Conv ---
    int8_t* buf__model_7_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_8_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_287 = alloc_buf(123904);
    cpu_pad(buf__model_7_act_Mul_output_0_DequantizeLinear_Output, pad_287, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_287_phy, buf__model_8_cv1_conv_Conv_output_0_phy, wt_287_phy, bias_287_phy, 22, 22, 256, 256, 3, 1, SCALE_287);
    cpu_silu(buf__model_8_cv1_conv_Conv_output_0, 102400, SCALE_287, ZP_287);

    // --- Node 295: Conv ---
    int8_t* buf__model_8_Split_output_1_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_8_m_0_cv1_conv_Conv_output_0 = alloc_buf(28800);
    int8_t* pad_295 = alloc_buf(61952);
    cpu_pad(buf__model_8_Split_output_1_DequantizeLinear_Output, pad_295, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(pad_295_phy, buf__model_8_m_0_cv1_conv_Conv_output_0_phy, wt_295_phy, bias_295_phy, 22, 22, 128, 72, 3, 1, SCALE_295);
    cpu_silu(buf__model_8_m_0_cv1_conv_Conv_output_0, 28800, SCALE_295, ZP_295);

    // --- Node 302: Conv ---
    int8_t* buf__model_8_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(28800);
    int8_t* buf__model_8_m_0_cv2_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_302 = alloc_buf(34848);
    cpu_pad(buf__model_8_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_302, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(pad_302_phy, buf__model_8_m_0_cv2_conv_Conv_output_0_phy, wt_302_phy, bias_302_phy, 22, 22, 72, 128, 3, 1, SCALE_302);
    cpu_silu(buf__model_8_m_0_cv2_conv_Conv_output_0, 51200, SCALE_302, ZP_302);

    // --- Node 308: Concat ---
    int8_t* buf__model_8_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_8_Split_output_0, 128, buf__model_8_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_8_Split_output_1_DequantizeLinear_Output, 128, buf__model_8_Concat_output_0, 128, 20, 20);
    cpu_concat_append(buf__model_8_m_0_Add_output_0, 128, buf__model_8_Concat_output_0, 256, 20, 20);

    // --- Node 311: Conv ---
    int8_t* buf__model_8_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_8_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_311 = alloc_buf(185856);
    cpu_pad(buf__model_8_Concat_output_0_DequantizeLinear_Output, pad_311, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_311_phy, buf__model_8_cv2_conv_Conv_output_0_phy, wt_311_phy, bias_311_phy, 22, 22, 384, 256, 3, 1, SCALE_311);
    cpu_silu(buf__model_8_cv2_conv_Conv_output_0, 102400, SCALE_311, ZP_311);

    // --- Node 318: Conv ---
    int8_t* buf__model_8_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_9_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_318 = alloc_buf(123904);
    cpu_pad(buf__model_8_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_318, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_318_phy, buf__model_9_cv1_conv_Conv_output_0_phy, wt_318_phy, bias_318_phy, 22, 22, 256, 128, 3, 1, SCALE_318);
    cpu_silu(buf__model_9_cv1_conv_Conv_output_0, 51200, SCALE_318, ZP_318);

    // --- Node 323: MaxPool ---
    int8_t* buf__model_9_m_MaxPool_output_0 = alloc_buf(51200);
    cpu_maxpool_5x5(buf__model_9_cv1_act_Mul_output_0, buf__model_9_m_MaxPool_output_0, 20, 20, 128);

    // --- Node 324: MaxPool ---
    int8_t* buf__model_9_m_1_MaxPool_output_0 = alloc_buf(51200);
    cpu_maxpool_5x5(buf__model_9_m_MaxPool_output_0, buf__model_9_m_1_MaxPool_output_0, 20, 20, 128);

    // --- Node 325: MaxPool ---
    int8_t* buf__model_9_m_2_MaxPool_output_0 = alloc_buf(51200);
    cpu_maxpool_5x5(buf__model_9_m_1_MaxPool_output_0, buf__model_9_m_2_MaxPool_output_0, 20, 20, 128);

    // --- Node 326: Concat ---
    int8_t* buf__model_9_Concat_output_0 = alloc_buf(204800);
    cpu_concat_append(buf__model_9_cv1_act_Mul_output_0, 128, buf__model_9_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_9_m_MaxPool_output_0, 128, buf__model_9_Concat_output_0, 128, 20, 20);
    cpu_concat_append(buf__model_9_m_1_MaxPool_output_0, 128, buf__model_9_Concat_output_0, 256, 20, 20);
    cpu_concat_append(buf__model_9_m_2_MaxPool_output_0, 128, buf__model_9_Concat_output_0, 384, 20, 20);

    // --- Node 329: Conv ---
    int8_t* buf__model_9_Concat_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_9_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_329 = alloc_buf(247808);
    cpu_pad(buf__model_9_Concat_output_0_DequantizeLinear_Output, pad_329, 20, 20, 512, INPUT_ZP);
    execute_fpga_layer(pad_329_phy, buf__model_9_cv2_conv_Conv_output_0_phy, wt_329_phy, bias_329_phy, 22, 22, 512, 256, 3, 1, SCALE_329);
    cpu_silu(buf__model_9_cv2_conv_Conv_output_0, 102400, SCALE_329, ZP_329);

    // --- Node 334: Upsample ---
    int8_t* buf__model_10_Resize_output_0 = alloc_buf(409600);
    cpu_upsample(buf__model_9_cv2_act_Mul_output_0, buf__model_10_Resize_output_0, 20, 20, 256, 2);

    // --- Node 335: Concat ---
    int8_t* buf__model_11_Concat_output_0 = alloc_buf(614400);
    cpu_concat_append(buf__model_10_Resize_output_0, 256, buf__model_11_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_6_cv2_act_Mul_output_0_DequantizeLinear_Output, 128, buf__model_11_Concat_output_0, 256, 40, 40);

    // --- Node 338: Conv ---
    int8_t* buf__model_11_Concat_output_0_DequantizeLinear_Output = alloc_buf(614400);
    int8_t* buf__model_12_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_338 = alloc_buf(677376);
    cpu_pad(buf__model_11_Concat_output_0_DequantizeLinear_Output, pad_338, 40, 40, 384, INPUT_ZP);
    execute_fpga_layer(pad_338_phy, buf__model_12_cv1_conv_Conv_output_0_phy, wt_338_phy, bias_338_phy, 42, 42, 384, 128, 3, 1, SCALE_338);
    cpu_silu(buf__model_12_cv1_conv_Conv_output_0, 204800, SCALE_338, ZP_338);

    // --- Node 346: Conv ---
    int8_t* buf__model_12_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_12_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_346 = alloc_buf(112896);
    cpu_pad(buf__model_12_Split_output_1_DequantizeLinear_Output, pad_346, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_346_phy, buf__model_12_m_0_cv1_conv_Conv_output_0_phy, wt_346_phy, bias_346_phy, 42, 42, 64, 32, 3, 1, SCALE_346);
    cpu_silu(buf__model_12_m_0_cv1_conv_Conv_output_0, 51200, SCALE_346, ZP_346);

    // --- Node 353: Conv ---
    int8_t* buf__model_12_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_12_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_353 = alloc_buf(56448);
    cpu_pad(buf__model_12_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_353, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_353_phy, buf__model_12_m_0_cv2_conv_Conv_output_0_phy, wt_353_phy, bias_353_phy, 42, 42, 32, 64, 3, 1, SCALE_353);
    cpu_silu(buf__model_12_m_0_cv2_conv_Conv_output_0, 102400, SCALE_353, ZP_353);

    // --- Node 358: Concat ---
    int8_t* buf__model_12_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_12_Split_output_0, 64, buf__model_12_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_12_Split_output_1_DequantizeLinear_Output, 64, buf__model_12_Concat_output_0, 64, 40, 40);
    cpu_concat_append(buf__model_12_m_0_cv2_act_Mul_output_0, 64, buf__model_12_Concat_output_0, 128, 40, 40);

    // --- Node 361: Conv ---
    int8_t* buf__model_12_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_12_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_361 = alloc_buf(338688);
    cpu_pad(buf__model_12_Concat_output_0_DequantizeLinear_Output, pad_361, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_361_phy, buf__model_12_cv2_conv_Conv_output_0_phy, wt_361_phy, bias_361_phy, 42, 42, 192, 128, 3, 1, SCALE_361);
    cpu_silu(buf__model_12_cv2_conv_Conv_output_0, 204800, SCALE_361, ZP_361);

    // --- Node 366: Upsample ---
    int8_t* buf__model_13_Resize_output_0 = alloc_buf(819200);
    cpu_upsample(buf__model_12_cv2_act_Mul_output_0, buf__model_13_Resize_output_0, 40, 40, 128, 2);

    // --- Node 367: Concat ---
    int8_t* buf__model_14_Concat_output_0 = alloc_buf(1228800);
    cpu_concat_append(buf__model_13_Resize_output_0, 128, buf__model_14_Concat_output_0, 0, 80, 80);
    cpu_concat_append(buf__model_4_cv2_act_Mul_output_0_DequantizeLinear_Output, 64, buf__model_14_Concat_output_0, 128, 80, 80);

    // --- Node 370: Conv ---
    int8_t* buf__model_14_Concat_output_0_DequantizeLinear_Output = alloc_buf(1228800);
    int8_t* buf__model_15_cv1_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_370 = alloc_buf(1291008);
    cpu_pad(buf__model_14_Concat_output_0_DequantizeLinear_Output, pad_370, 80, 80, 192, INPUT_ZP);
    execute_fpga_layer(pad_370_phy, buf__model_15_cv1_conv_Conv_output_0_phy, wt_370_phy, bias_370_phy, 82, 82, 192, 64, 3, 1, SCALE_370);
    cpu_silu(buf__model_15_cv1_conv_Conv_output_0, 409600, SCALE_370, ZP_370);

    // --- Node 378: Conv ---
    int8_t* buf__model_15_Split_output_1_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_15_m_0_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_378 = alloc_buf(215168);
    cpu_pad(buf__model_15_Split_output_1_DequantizeLinear_Output, pad_378, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_378_phy, buf__model_15_m_0_cv1_conv_Conv_output_0_phy, wt_378_phy, bias_378_phy, 82, 82, 32, 16, 3, 1, SCALE_378);
    cpu_silu(buf__model_15_m_0_cv1_conv_Conv_output_0, 102400, SCALE_378, ZP_378);

    // --- Node 385: Conv ---
    int8_t* buf__model_15_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_15_m_0_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_385 = alloc_buf(107584);
    cpu_pad(buf__model_15_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_385, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_385_phy, buf__model_15_m_0_cv2_conv_Conv_output_0_phy, wt_385_phy, bias_385_phy, 82, 82, 16, 32, 3, 1, SCALE_385);
    cpu_silu(buf__model_15_m_0_cv2_conv_Conv_output_0, 204800, SCALE_385, ZP_385);

    // --- Node 390: Concat ---
    int8_t* buf__model_15_Concat_output_0 = alloc_buf(614400);
    cpu_concat_append(buf__model_15_Split_output_0, 32, buf__model_15_Concat_output_0, 0, 80, 80);
    cpu_concat_append(buf__model_15_Split_output_1_DequantizeLinear_Output, 32, buf__model_15_Concat_output_0, 32, 80, 80);
    cpu_concat_append(buf__model_15_m_0_cv2_act_Mul_output_0, 32, buf__model_15_Concat_output_0, 64, 80, 80);

    // --- Node 393: Conv ---
    int8_t* buf__model_15_Concat_output_0_DequantizeLinear_Output = alloc_buf(614400);
    int8_t* buf__model_15_cv2_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_393 = alloc_buf(645504);
    cpu_pad(buf__model_15_Concat_output_0_DequantizeLinear_Output, pad_393, 80, 80, 96, INPUT_ZP);
    execute_fpga_layer(pad_393_phy, buf__model_15_cv2_conv_Conv_output_0_phy, wt_393_phy, bias_393_phy, 82, 82, 96, 64, 3, 1, SCALE_393);
    cpu_silu(buf__model_15_cv2_conv_Conv_output_0, 409600, SCALE_393, ZP_393);

    // --- Node 400: Conv ---
    int8_t* buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_16_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_400 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_400, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_400_phy, buf__model_16_conv_Conv_output_0_phy, wt_400_phy, bias_400_phy, 82, 82, 64, 64, 3, 2, SCALE_400);
    cpu_silu(buf__model_16_conv_Conv_output_0, 102400, SCALE_400, ZP_400);

    // --- Node 401: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_401 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_401, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_401_phy, buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0_phy, wt_401_phy, bias_401_phy, 82, 82, 64, 64, 3, 1, SCALE_401);
    cpu_silu(buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0, 409600, SCALE_401, ZP_401);

    // --- Node 402: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_402 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_402, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_402_phy, buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0_phy, wt_402_phy, bias_402_phy, 82, 82, 64, 80, 3, 1, SCALE_402);
    cpu_silu(buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0, 512000, SCALE_402, ZP_402);

    // --- Node 415: Concat ---
    int8_t* buf__model_17_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_16_act_Mul_output_0, 64, buf__model_17_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_12_cv2_act_Mul_output_0, 128, buf__model_17_Concat_output_0, 64, 40, 40);

    // --- Node 422: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_422 = alloc_buf(430336);
    cpu_pad(buf__model_22_cv2_0_cv2_0_0_act_Mul_output_0_DequantizeLinear_Output, pad_422, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_422_phy, buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0_phy, wt_422_phy, bias_422_phy, 82, 82, 64, 64, 3, 1, SCALE_422);
    cpu_silu(buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0, 409600, SCALE_422, ZP_422);

    // --- Node 423: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(512000);
    int8_t* buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_423 = alloc_buf(537920);
    cpu_pad(buf__model_22_cv3_0_cv3_0_0_act_Mul_output_0_DequantizeLinear_Output, pad_423, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(pad_423_phy, buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0_phy, wt_423_phy, bias_423_phy, 82, 82, 80, 80, 3, 1, SCALE_423);
    cpu_silu(buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0, 512000, SCALE_423, ZP_423);

    // --- Node 424: Conv ---
    int8_t* buf__model_17_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_18_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_424 = alloc_buf(338688);
    cpu_pad(buf__model_17_Concat_output_0_DequantizeLinear_Output, pad_424, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_424_phy, buf__model_18_cv1_conv_Conv_output_0_phy, wt_424_phy, bias_424_phy, 42, 42, 192, 128, 3, 1, SCALE_424);
    cpu_silu(buf__model_18_cv1_conv_Conv_output_0, 204800, SCALE_424, ZP_424);

    // --- Node 443: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_22_cv2_0_cv2_0_2_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_443 = alloc_buf(430336);
    cpu_pad(buf__model_22_cv2_0_cv2_0_1_act_Mul_output_0_DequantizeLinear_Output, pad_443, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_443_phy, buf__model_22_cv2_0_cv2_0_2_Conv_output_0_phy, wt_443_phy, bias_443_phy, 82, 82, 64, 64, 3, 1, SCALE_443);
    cpu_silu(buf__model_22_cv2_0_cv2_0_2_Conv_output_0, 409600, SCALE_443, ZP_443);

    // --- Node 444: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(512000);
    int8_t* buf__model_22_cv3_0_cv3_0_2_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_444 = alloc_buf(537920);
    cpu_pad(buf__model_22_cv3_0_cv3_0_1_act_Mul_output_0_DequantizeLinear_Output, pad_444, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(pad_444_phy, buf__model_22_cv3_0_cv3_0_2_Conv_output_0_phy, wt_444_phy, bias_444_phy, 82, 82, 80, 80, 3, 1, SCALE_444);
    cpu_silu(buf__model_22_cv3_0_cv3_0_2_Conv_output_0, 512000, SCALE_444, ZP_444);

    // --- Node 448: Conv ---
    int8_t* buf__model_18_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_18_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_448 = alloc_buf(112896);
    cpu_pad(buf__model_18_Split_output_1_DequantizeLinear_Output, pad_448, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_448_phy, buf__model_18_m_0_cv1_conv_Conv_output_0_phy, wt_448_phy, bias_448_phy, 42, 42, 64, 32, 3, 1, SCALE_448);
    cpu_silu(buf__model_18_m_0_cv1_conv_Conv_output_0, 51200, SCALE_448, ZP_448);

    // --- Node 459: Conv ---
    int8_t* buf__model_18_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_18_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_459 = alloc_buf(56448);
    cpu_pad(buf__model_18_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_459, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_459_phy, buf__model_18_m_0_cv2_conv_Conv_output_0_phy, wt_459_phy, bias_459_phy, 42, 42, 32, 64, 3, 1, SCALE_459);
    cpu_silu(buf__model_18_m_0_cv2_conv_Conv_output_0, 102400, SCALE_459, ZP_459);

    // --- Node 464: Concat ---
    int8_t* buf__model_18_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_18_Split_output_0, 64, buf__model_18_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_18_Split_output_1_DequantizeLinear_Output, 64, buf__model_18_Concat_output_0, 64, 40, 40);
    cpu_concat_append(buf__model_18_m_0_cv2_act_Mul_output_0, 64, buf__model_18_Concat_output_0, 128, 40, 40);

    // --- Node 467: Conv ---
    int8_t* buf__model_18_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_18_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_467 = alloc_buf(338688);
    cpu_pad(buf__model_18_Concat_output_0_DequantizeLinear_Output, pad_467, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_467_phy, buf__model_18_cv2_conv_Conv_output_0_phy, wt_467_phy, bias_467_phy, 42, 42, 192, 128, 3, 1, SCALE_467);
    cpu_silu(buf__model_18_cv2_conv_Conv_output_0, 204800, SCALE_467, ZP_467);

    // --- Node 474: Conv ---
    int8_t* buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_19_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_474 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_474, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_474_phy, buf__model_19_conv_Conv_output_0_phy, wt_474_phy, bias_474_phy, 42, 42, 128, 128, 3, 2, SCALE_474);
    cpu_silu(buf__model_19_conv_Conv_output_0, 51200, SCALE_474, ZP_474);

    // --- Node 475: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_475 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_475, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_475_phy, buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0_phy, wt_475_phy, bias_475_phy, 42, 42, 128, 64, 3, 1, SCALE_475);
    cpu_silu(buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0, 102400, SCALE_475, ZP_475);

    // --- Node 476: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_476 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_476, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_476_phy, buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0_phy, wt_476_phy, bias_476_phy, 42, 42, 128, 80, 3, 1, SCALE_476);
    cpu_silu(buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0, 128000, SCALE_476, ZP_476);

    // --- Node 489: Concat ---
    int8_t* buf__model_20_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_19_act_Mul_output_0, 128, buf__model_20_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_9_cv2_act_Mul_output_0, 256, buf__model_20_Concat_output_0, 128, 20, 20);

    // --- Node 496: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_496 = alloc_buf(112896);
    cpu_pad(buf__model_22_cv2_1_cv2_1_0_act_Mul_output_0_DequantizeLinear_Output, pad_496, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_496_phy, buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0_phy, wt_496_phy, bias_496_phy, 42, 42, 64, 64, 3, 1, SCALE_496);
    cpu_silu(buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0, 102400, SCALE_496, ZP_496);

    // --- Node 497: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(128000);
    int8_t* buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_497 = alloc_buf(141120);
    cpu_pad(buf__model_22_cv3_1_cv3_1_0_act_Mul_output_0_DequantizeLinear_Output, pad_497, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(pad_497_phy, buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0_phy, wt_497_phy, bias_497_phy, 42, 42, 80, 80, 3, 1, SCALE_497);
    cpu_silu(buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0, 128000, SCALE_497, ZP_497);

    // --- Node 498: Conv ---
    int8_t* buf__model_20_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_21_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_498 = alloc_buf(185856);
    cpu_pad(buf__model_20_Concat_output_0_DequantizeLinear_Output, pad_498, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_498_phy, buf__model_21_cv1_conv_Conv_output_0_phy, wt_498_phy, bias_498_phy, 22, 22, 384, 256, 3, 1, SCALE_498);
    cpu_silu(buf__model_21_cv1_conv_Conv_output_0, 102400, SCALE_498, ZP_498);

    // --- Node 517: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_1_cv2_1_2_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_517 = alloc_buf(112896);
    cpu_pad(buf__model_22_cv2_1_cv2_1_1_act_Mul_output_0_DequantizeLinear_Output, pad_517, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_517_phy, buf__model_22_cv2_1_cv2_1_2_Conv_output_0_phy, wt_517_phy, bias_517_phy, 42, 42, 64, 64, 3, 1, SCALE_517);
    cpu_silu(buf__model_22_cv2_1_cv2_1_2_Conv_output_0, 102400, SCALE_517, ZP_517);

    // --- Node 518: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(128000);
    int8_t* buf__model_22_cv3_1_cv3_1_2_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_518 = alloc_buf(141120);
    cpu_pad(buf__model_22_cv3_1_cv3_1_1_act_Mul_output_0_DequantizeLinear_Output, pad_518, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(pad_518_phy, buf__model_22_cv3_1_cv3_1_2_Conv_output_0_phy, wt_518_phy, bias_518_phy, 42, 42, 80, 80, 3, 1, SCALE_518);
    cpu_silu(buf__model_22_cv3_1_cv3_1_2_Conv_output_0, 128000, SCALE_518, ZP_518);

    // --- Node 522: Conv ---
    int8_t* buf__model_21_Split_output_1_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_21_m_0_cv1_conv_Conv_output_0 = alloc_buf(28800);
    int8_t* pad_522 = alloc_buf(61952);
    cpu_pad(buf__model_21_Split_output_1_DequantizeLinear_Output, pad_522, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(pad_522_phy, buf__model_21_m_0_cv1_conv_Conv_output_0_phy, wt_522_phy, bias_522_phy, 22, 22, 128, 72, 3, 1, SCALE_522);
    cpu_silu(buf__model_21_m_0_cv1_conv_Conv_output_0, 28800, SCALE_522, ZP_522);

    // --- Node 533: Conv ---
    int8_t* buf__model_21_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(28800);
    int8_t* buf__model_21_m_0_cv2_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_533 = alloc_buf(34848);
    cpu_pad(buf__model_21_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_533, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(pad_533_phy, buf__model_21_m_0_cv2_conv_Conv_output_0_phy, wt_533_phy, bias_533_phy, 22, 22, 72, 128, 3, 1, SCALE_533);
    cpu_silu(buf__model_21_m_0_cv2_conv_Conv_output_0, 51200, SCALE_533, ZP_533);

    // --- Node 538: Concat ---
    int8_t* buf__model_21_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_21_Split_output_0, 128, buf__model_21_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_21_Split_output_1_DequantizeLinear_Output, 128, buf__model_21_Concat_output_0, 128, 20, 20);
    cpu_concat_append(buf__model_21_m_0_cv2_act_Mul_output_0, 128, buf__model_21_Concat_output_0, 256, 20, 20);

    // --- Node 541: Conv ---
    int8_t* buf__model_21_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_21_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_541 = alloc_buf(185856);
    cpu_pad(buf__model_21_Concat_output_0_DequantizeLinear_Output, pad_541, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_541_phy, buf__model_21_cv2_conv_Conv_output_0_phy, wt_541_phy, bias_541_phy, 22, 22, 384, 256, 3, 1, SCALE_541);
    cpu_silu(buf__model_21_cv2_conv_Conv_output_0, 102400, SCALE_541, ZP_541);

    // --- Node 548: Conv ---
    int8_t* buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_548 = alloc_buf(123904);
    cpu_pad(buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_548, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_548_phy, buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0_phy, wt_548_phy, bias_548_phy, 22, 22, 256, 64, 3, 1, SCALE_548);
    cpu_silu(buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0, 25600, SCALE_548, ZP_548);

    // --- Node 549: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_549 = alloc_buf(123904);
    cpu_pad(buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_549, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_549_phy, buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0_phy, wt_549_phy, bias_549_phy, 22, 22, 256, 80, 3, 1, SCALE_549);
    cpu_silu(buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0, 32000, SCALE_549, ZP_549);

    // --- Node 562: Conv ---
    int8_t* buf__model_22_cv2_2_cv2_2_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(25600);
    int8_t* buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_562 = alloc_buf(30976);
    cpu_pad(buf__model_22_cv2_2_cv2_2_0_act_Mul_output_0_DequantizeLinear_Output, pad_562, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(pad_562_phy, buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0_phy, wt_562_phy, bias_562_phy, 22, 22, 64, 64, 3, 1, SCALE_562);
    cpu_silu(buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0, 25600, SCALE_562, ZP_562);

    // --- Node 563: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(32000);
    int8_t* buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_563 = alloc_buf(38720);
    cpu_pad(buf__model_22_cv3_2_cv3_2_0_act_Mul_output_0_DequantizeLinear_Output, pad_563, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(pad_563_phy, buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0_phy, wt_563_phy, bias_563_phy, 22, 22, 80, 80, 3, 1, SCALE_563);
    cpu_silu(buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0, 32000, SCALE_563, ZP_563);

    // --- Node 576: Conv ---
    int8_t* buf__model_22_cv2_2_cv2_2_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(25600);
    int8_t* buf__model_22_cv2_2_cv2_2_2_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_576 = alloc_buf(30976);
    cpu_pad(buf__model_22_cv2_2_cv2_2_1_act_Mul_output_0_DequantizeLinear_Output, pad_576, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(pad_576_phy, buf__model_22_cv2_2_cv2_2_2_Conv_output_0_phy, wt_576_phy, bias_576_phy, 22, 22, 64, 64, 3, 1, SCALE_576);
    cpu_silu(buf__model_22_cv2_2_cv2_2_2_Conv_output_0, 25600, SCALE_576, ZP_576);

    // --- Node 577: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(32000);
    int8_t* buf__model_22_cv3_2_cv3_2_2_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_577 = alloc_buf(38720);
    cpu_pad(buf__model_22_cv3_2_cv3_2_1_act_Mul_output_0_DequantizeLinear_Output, pad_577, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(pad_577_phy, buf__model_22_cv3_2_cv3_2_2_Conv_output_0_phy, wt_577_phy, bias_577_phy, 22, 22, 80, 80, 3, 1, SCALE_577);
    cpu_silu(buf__model_22_cv3_2_cv3_2_2_Conv_output_0, 32000, SCALE_577, ZP_577);

    // --- Node 591: Conv ---
    int8_t* buf__model_22_dfl_Transpose_1_output_0_DequantizeLinear_Output = alloc_buf(537600);
    int8_t* buf__model_22_dfl_conv_Conv_output_0 = alloc_buf(33600);
    int8_t* pad_591 = alloc_buf(806592);
    cpu_pad(buf__model_22_dfl_Transpose_1_output_0_DequantizeLinear_Output, pad_591, 4, 8400, 16, INPUT_ZP);
    execute_fpga_layer(pad_591_phy, buf__model_22_dfl_conv_Conv_output_0_phy, wt_591_phy, bias_591_phy, 6, 8402, 16, 1, 3, 1, SCALE_591);
    cpu_silu(buf__model_22_dfl_conv_Conv_output_0, 33600, SCALE_591, ZP_591);

    std::cout << "Total CMA Memory Used: " << current_offset / (1024*1024) << " MB" << std::endl;
}
