void run_network() {
    // --- AUTO-GENERATED FPGA SCHEDULER ---

    // --- Layer 0 (/model.0/conv/Conv) ---
    // Input: 640x640x3 | Output Channels: 16 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_0_in, buf_0_pad, 640, 640, 3, INPUT_ZP);
    execute_fpga_layer(buf_0_pad_phy, buf_0_out_phy, wt_0_phy, bias_0_phy, 642, 642, 3, 16, 3, 2, LAYER_0_SCALE);
    cpu_silu(buf_0_out_virt, 320 * 320 * 16, LAYER_0_SCALE, LAYER_0_ZP);

    // --- Layer 1 (/model.1/conv/Conv) ---
    // Input: 320x320x16 | Output Channels: 32 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_1_in, buf_1_pad, 320, 320, 16, INPUT_ZP);
    execute_fpga_layer(buf_1_pad_phy, buf_1_out_phy, wt_1_phy, bias_1_phy, 322, 322, 16, 32, 3, 2, LAYER_1_SCALE);
    cpu_silu(buf_1_out_virt, 160 * 160 * 32, LAYER_1_SCALE, LAYER_1_ZP);

    // --- Layer 2 (/model.2/cv1/conv/Conv) ---
    // Input: 160x160x32 | Output Channels: 32 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_2_in, buf_2_pad, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(buf_2_pad_phy, buf_2_out_phy, wt_2_phy, bias_2_phy, 162, 162, 32, 32, 1, 1, LAYER_2_SCALE);
    cpu_silu(buf_2_out_virt, 160 * 160 * 32, LAYER_2_SCALE, LAYER_2_ZP);

    // --- Layer 3 (/model.2/m.0/cv1/conv/Conv) ---
    // Input: 160x160x16 | Output Channels: 8 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_3_in, buf_3_pad, 160, 160, 16, INPUT_ZP);
    execute_fpga_layer(buf_3_pad_phy, buf_3_out_phy, wt_3_phy, bias_3_phy, 162, 162, 16, 8, 3, 1, LAYER_3_SCALE);
    cpu_silu(buf_3_out_virt, 160 * 160 * 8, LAYER_3_SCALE, LAYER_3_ZP);

    // --- Layer 4 (/model.2/m.0/cv2/conv/Conv) ---
    // Input: 160x160x8 | Output Channels: 16 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_4_in, buf_4_pad, 160, 160, 8, INPUT_ZP);
    execute_fpga_layer(buf_4_pad_phy, buf_4_out_phy, wt_4_phy, bias_4_phy, 162, 162, 8, 16, 3, 1, LAYER_4_SCALE);
    cpu_silu(buf_4_out_virt, 160 * 160 * 16, LAYER_4_SCALE, LAYER_4_ZP);

    // --- Layer 5 (/model.2/cv2/conv/Conv) ---
    // Input: 160x160x48 | Output Channels: 32 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_5_in, buf_5_pad, 160, 160, 48, INPUT_ZP);
    execute_fpga_layer(buf_5_pad_phy, buf_5_out_phy, wt_5_phy, bias_5_phy, 162, 162, 48, 32, 1, 1, LAYER_5_SCALE);
    cpu_silu(buf_5_out_virt, 160 * 160 * 32, LAYER_5_SCALE, LAYER_5_ZP);

    // --- Layer 6 (/model.3/conv/Conv) ---
    // Input: 160x160x32 | Output Channels: 64 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_6_in, buf_6_pad, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(buf_6_pad_phy, buf_6_out_phy, wt_6_phy, bias_6_phy, 162, 162, 32, 64, 3, 2, LAYER_6_SCALE);
    cpu_silu(buf_6_out_virt, 80 * 80 * 64, LAYER_6_SCALE, LAYER_6_ZP);

    // --- Layer 7 (/model.4/cv1/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_7_in, buf_7_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_7_pad_phy, buf_7_out_phy, wt_7_phy, bias_7_phy, 82, 82, 64, 64, 1, 1, LAYER_7_SCALE);
    cpu_silu(buf_7_out_virt, 80 * 80 * 64, LAYER_7_SCALE, LAYER_7_ZP);

    // --- Layer 8 (/model.4/m.0/cv1/conv/Conv) ---
    // Input: 80x80x32 | Output Channels: 16 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_8_in, buf_8_pad, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(buf_8_pad_phy, buf_8_out_phy, wt_8_phy, bias_8_phy, 82, 82, 32, 16, 3, 1, LAYER_8_SCALE);
    cpu_silu(buf_8_out_virt, 80 * 80 * 16, LAYER_8_SCALE, LAYER_8_ZP);

    // --- Layer 9 (/model.4/m.0/cv2/conv/Conv) ---
    // Input: 80x80x16 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_9_in, buf_9_pad, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(buf_9_pad_phy, buf_9_out_phy, wt_9_phy, bias_9_phy, 82, 82, 16, 32, 3, 1, LAYER_9_SCALE);
    cpu_silu(buf_9_out_virt, 80 * 80 * 32, LAYER_9_SCALE, LAYER_9_ZP);

    // --- Layer 10 (/model.4/m.1/cv1/conv/Conv) ---
    // Input: 80x80x32 | Output Channels: 16 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_10_in, buf_10_pad, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(buf_10_pad_phy, buf_10_out_phy, wt_10_phy, bias_10_phy, 82, 82, 32, 16, 3, 1, LAYER_10_SCALE);
    cpu_silu(buf_10_out_virt, 80 * 80 * 16, LAYER_10_SCALE, LAYER_10_ZP);

    // --- Layer 11 (/model.4/m.1/cv2/conv/Conv) ---
    // Input: 80x80x16 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_11_in, buf_11_pad, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(buf_11_pad_phy, buf_11_out_phy, wt_11_phy, bias_11_phy, 82, 82, 16, 32, 3, 1, LAYER_11_SCALE);
    cpu_silu(buf_11_out_virt, 80 * 80 * 32, LAYER_11_SCALE, LAYER_11_ZP);

    // --- Layer 12 (/model.4/cv2/conv/Conv) ---
    // Input: 80x80x128 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_12_in, buf_12_pad, 80, 80, 128, INPUT_ZP);
    execute_fpga_layer(buf_12_pad_phy, buf_12_out_phy, wt_12_phy, bias_12_phy, 82, 82, 128, 64, 1, 1, LAYER_12_SCALE);
    cpu_silu(buf_12_out_virt, 80 * 80 * 64, LAYER_12_SCALE, LAYER_12_ZP);

    // --- Layer 13 (/model.5/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 128 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_13_in, buf_13_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_13_pad_phy, buf_13_out_phy, wt_13_phy, bias_13_phy, 82, 82, 64, 128, 3, 2, LAYER_13_SCALE);
    cpu_silu(buf_13_out_virt, 40 * 40 * 128, LAYER_13_SCALE, LAYER_13_ZP);

    // --- Layer 14 (/model.6/cv1/conv/Conv) ---
    // Input: 40x40x128 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_14_in, buf_14_pad, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(buf_14_pad_phy, buf_14_out_phy, wt_14_phy, bias_14_phy, 42, 42, 128, 128, 1, 1, LAYER_14_SCALE);
    cpu_silu(buf_14_out_virt, 40 * 40 * 128, LAYER_14_SCALE, LAYER_14_ZP);

    // --- Layer 15 (/model.6/m.0/cv1/conv/Conv) ---
    // Input: 40x40x64 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_15_in, buf_15_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_15_pad_phy, buf_15_out_phy, wt_15_phy, bias_15_phy, 42, 42, 64, 32, 3, 1, LAYER_15_SCALE);
    cpu_silu(buf_15_out_virt, 40 * 40 * 32, LAYER_15_SCALE, LAYER_15_ZP);

    // --- Layer 16 (/model.6/m.0/cv2/conv/Conv) ---
    // Input: 40x40x32 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_16_in, buf_16_pad, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(buf_16_pad_phy, buf_16_out_phy, wt_16_phy, bias_16_phy, 42, 42, 32, 64, 3, 1, LAYER_16_SCALE);
    cpu_silu(buf_16_out_virt, 40 * 40 * 64, LAYER_16_SCALE, LAYER_16_ZP);

    // --- Layer 17 (/model.6/m.1/cv1/conv/Conv) ---
    // Input: 40x40x64 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_17_in, buf_17_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_17_pad_phy, buf_17_out_phy, wt_17_phy, bias_17_phy, 42, 42, 64, 32, 3, 1, LAYER_17_SCALE);
    cpu_silu(buf_17_out_virt, 40 * 40 * 32, LAYER_17_SCALE, LAYER_17_ZP);

    // --- Layer 18 (/model.6/m.1/cv2/conv/Conv) ---
    // Input: 40x40x32 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_18_in, buf_18_pad, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(buf_18_pad_phy, buf_18_out_phy, wt_18_phy, bias_18_phy, 42, 42, 32, 64, 3, 1, LAYER_18_SCALE);
    cpu_silu(buf_18_out_virt, 40 * 40 * 64, LAYER_18_SCALE, LAYER_18_ZP);

    // --- Layer 19 (/model.6/cv2/conv/Conv) ---
    // Input: 40x40x256 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_19_in, buf_19_pad, 40, 40, 256, INPUT_ZP);
    execute_fpga_layer(buf_19_pad_phy, buf_19_out_phy, wt_19_phy, bias_19_phy, 42, 42, 256, 128, 1, 1, LAYER_19_SCALE);
    cpu_silu(buf_19_out_virt, 40 * 40 * 128, LAYER_19_SCALE, LAYER_19_ZP);

    // --- Layer 20 (/model.7/conv/Conv) ---
    // Input: 40x40x128 | Output Channels: 256 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_20_in, buf_20_pad, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(buf_20_pad_phy, buf_20_out_phy, wt_20_phy, bias_20_phy, 42, 42, 128, 256, 3, 2, LAYER_20_SCALE);
    cpu_silu(buf_20_out_virt, 20 * 20 * 256, LAYER_20_SCALE, LAYER_20_ZP);

    // --- Layer 21 (/model.8/cv1/conv/Conv) ---
    // Input: 20x20x256 | Output Channels: 256 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_21_in, buf_21_pad, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(buf_21_pad_phy, buf_21_out_phy, wt_21_phy, bias_21_phy, 22, 22, 256, 256, 1, 1, LAYER_21_SCALE);
    cpu_silu(buf_21_out_virt, 20 * 20 * 256, LAYER_21_SCALE, LAYER_21_ZP);

    // --- Layer 22 (/model.8/m.0/cv1/conv/Conv) ---
    // Input: 20x20x128 | Output Channels: 72 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_22_in, buf_22_pad, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(buf_22_pad_phy, buf_22_out_phy, wt_22_phy, bias_22_phy, 22, 22, 128, 72, 3, 1, LAYER_22_SCALE);
    cpu_silu(buf_22_out_virt, 20 * 20 * 72, LAYER_22_SCALE, LAYER_22_ZP);

    // --- Layer 23 (/model.8/m.0/cv2/conv/Conv) ---
    // Input: 20x20x72 | Output Channels: 128 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_23_in, buf_23_pad, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(buf_23_pad_phy, buf_23_out_phy, wt_23_phy, bias_23_phy, 22, 22, 72, 128, 3, 1, LAYER_23_SCALE);
    cpu_silu(buf_23_out_virt, 20 * 20 * 128, LAYER_23_SCALE, LAYER_23_ZP);

    // --- Layer 24 (/model.8/cv2/conv/Conv) ---
    // Input: 20x20x384 | Output Channels: 256 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_24_in, buf_24_pad, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(buf_24_pad_phy, buf_24_out_phy, wt_24_phy, bias_24_phy, 22, 22, 384, 256, 1, 1, LAYER_24_SCALE);
    cpu_silu(buf_24_out_virt, 20 * 20 * 256, LAYER_24_SCALE, LAYER_24_ZP);

    // --- Layer 25 (/model.9/cv1/conv/Conv) ---
    // Input: 20x20x256 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_25_in, buf_25_pad, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(buf_25_pad_phy, buf_25_out_phy, wt_25_phy, bias_25_phy, 22, 22, 256, 128, 1, 1, LAYER_25_SCALE);
    cpu_silu(buf_25_out_virt, 20 * 20 * 128, LAYER_25_SCALE, LAYER_25_ZP);

    // --- Layer 26 (/model.9/cv2/conv/Conv) ---
    // Input: 20x20x512 | Output Channels: 256 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_26_in, buf_26_pad, 20, 20, 512, INPUT_ZP);
    execute_fpga_layer(buf_26_pad_phy, buf_26_out_phy, wt_26_phy, bias_26_phy, 22, 22, 512, 256, 1, 1, LAYER_26_SCALE);
    cpu_silu(buf_26_out_virt, 20 * 20 * 256, LAYER_26_SCALE, LAYER_26_ZP);

    // --- Layer 27 (/model.12/cv1/conv/Conv) ---
    // Input: 40x40x384 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_27_in, buf_27_pad, 40, 40, 384, INPUT_ZP);
    execute_fpga_layer(buf_27_pad_phy, buf_27_out_phy, wt_27_phy, bias_27_phy, 42, 42, 384, 128, 1, 1, LAYER_27_SCALE);
    cpu_silu(buf_27_out_virt, 40 * 40 * 128, LAYER_27_SCALE, LAYER_27_ZP);

    // --- Layer 28 (/model.12/m.0/cv1/conv/Conv) ---
    // Input: 40x40x64 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_28_in, buf_28_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_28_pad_phy, buf_28_out_phy, wt_28_phy, bias_28_phy, 42, 42, 64, 32, 3, 1, LAYER_28_SCALE);
    cpu_silu(buf_28_out_virt, 40 * 40 * 32, LAYER_28_SCALE, LAYER_28_ZP);

    // --- Layer 29 (/model.12/m.0/cv2/conv/Conv) ---
    // Input: 40x40x32 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_29_in, buf_29_pad, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(buf_29_pad_phy, buf_29_out_phy, wt_29_phy, bias_29_phy, 42, 42, 32, 64, 3, 1, LAYER_29_SCALE);
    cpu_silu(buf_29_out_virt, 40 * 40 * 64, LAYER_29_SCALE, LAYER_29_ZP);

    // --- Layer 30 (/model.12/cv2/conv/Conv) ---
    // Input: 40x40x192 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_30_in, buf_30_pad, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(buf_30_pad_phy, buf_30_out_phy, wt_30_phy, bias_30_phy, 42, 42, 192, 128, 1, 1, LAYER_30_SCALE);
    cpu_silu(buf_30_out_virt, 40 * 40 * 128, LAYER_30_SCALE, LAYER_30_ZP);

    // --- Layer 31 (/model.15/cv1/conv/Conv) ---
    // Input: 80x80x192 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_31_in, buf_31_pad, 80, 80, 192, INPUT_ZP);
    execute_fpga_layer(buf_31_pad_phy, buf_31_out_phy, wt_31_phy, bias_31_phy, 82, 82, 192, 64, 1, 1, LAYER_31_SCALE);
    cpu_silu(buf_31_out_virt, 80 * 80 * 64, LAYER_31_SCALE, LAYER_31_ZP);

    // --- Layer 32 (/model.15/m.0/cv1/conv/Conv) ---
    // Input: 80x80x32 | Output Channels: 16 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_32_in, buf_32_pad, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(buf_32_pad_phy, buf_32_out_phy, wt_32_phy, bias_32_phy, 82, 82, 32, 16, 3, 1, LAYER_32_SCALE);
    cpu_silu(buf_32_out_virt, 80 * 80 * 16, LAYER_32_SCALE, LAYER_32_ZP);

    // --- Layer 33 (/model.15/m.0/cv2/conv/Conv) ---
    // Input: 80x80x16 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_33_in, buf_33_pad, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(buf_33_pad_phy, buf_33_out_phy, wt_33_phy, bias_33_phy, 82, 82, 16, 32, 3, 1, LAYER_33_SCALE);
    cpu_silu(buf_33_out_virt, 80 * 80 * 32, LAYER_33_SCALE, LAYER_33_ZP);

    // --- Layer 34 (/model.15/cv2/conv/Conv) ---
    // Input: 80x80x96 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_34_in, buf_34_pad, 80, 80, 96, INPUT_ZP);
    execute_fpga_layer(buf_34_pad_phy, buf_34_out_phy, wt_34_phy, bias_34_phy, 82, 82, 96, 64, 1, 1, LAYER_34_SCALE);
    cpu_silu(buf_34_out_virt, 80 * 80 * 64, LAYER_34_SCALE, LAYER_34_ZP);

    // --- Layer 35 (/model.16/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 64 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_35_in, buf_35_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_35_pad_phy, buf_35_out_phy, wt_35_phy, bias_35_phy, 82, 82, 64, 64, 3, 2, LAYER_35_SCALE);
    cpu_silu(buf_35_out_virt, 40 * 40 * 64, LAYER_35_SCALE, LAYER_35_ZP);

    // --- Layer 36 (/model.22/cv2.0/cv2.0.0/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_36_in, buf_36_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_36_pad_phy, buf_36_out_phy, wt_36_phy, bias_36_phy, 82, 82, 64, 64, 3, 1, LAYER_36_SCALE);
    cpu_silu(buf_36_out_virt, 80 * 80 * 64, LAYER_36_SCALE, LAYER_36_ZP);

    // --- Layer 37 (/model.22/cv3.0/cv3.0.0/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_37_in, buf_37_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_37_pad_phy, buf_37_out_phy, wt_37_phy, bias_37_phy, 82, 82, 64, 80, 3, 1, LAYER_37_SCALE);
    cpu_silu(buf_37_out_virt, 80 * 80 * 80, LAYER_37_SCALE, LAYER_37_ZP);

    // --- Layer 38 (/model.22/cv2.0/cv2.0.1/conv/Conv) ---
    // Input: 80x80x64 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_38_in, buf_38_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_38_pad_phy, buf_38_out_phy, wt_38_phy, bias_38_phy, 82, 82, 64, 64, 3, 1, LAYER_38_SCALE);
    cpu_silu(buf_38_out_virt, 80 * 80 * 64, LAYER_38_SCALE, LAYER_38_ZP);

    // --- Layer 39 (/model.22/cv3.0/cv3.0.1/conv/Conv) ---
    // Input: 80x80x80 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_39_in, buf_39_pad, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(buf_39_pad_phy, buf_39_out_phy, wt_39_phy, bias_39_phy, 82, 82, 80, 80, 3, 1, LAYER_39_SCALE);
    cpu_silu(buf_39_out_virt, 80 * 80 * 80, LAYER_39_SCALE, LAYER_39_ZP);

    // --- Layer 40 (/model.18/cv1/conv/Conv) ---
    // Input: 40x40x192 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_40_in, buf_40_pad, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(buf_40_pad_phy, buf_40_out_phy, wt_40_phy, bias_40_phy, 42, 42, 192, 128, 1, 1, LAYER_40_SCALE);
    cpu_silu(buf_40_out_virt, 40 * 40 * 128, LAYER_40_SCALE, LAYER_40_ZP);

    // --- Layer 41 (/model.22/cv2.0/cv2.0.2/Conv) ---
    // Input: 80x80x64 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_41_in, buf_41_pad, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(buf_41_pad_phy, buf_41_out_phy, wt_41_phy, bias_41_phy, 82, 82, 64, 64, 1, 1, LAYER_41_SCALE);
    cpu_silu(buf_41_out_virt, 80 * 80 * 64, LAYER_41_SCALE, LAYER_41_ZP);

    // --- Layer 42 (/model.22/cv3.0/cv3.0.2/Conv) ---
    // Input: 80x80x80 | Output Channels: 80 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_42_in, buf_42_pad, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(buf_42_pad_phy, buf_42_out_phy, wt_42_phy, bias_42_phy, 82, 82, 80, 80, 1, 1, LAYER_42_SCALE);
    cpu_silu(buf_42_out_virt, 80 * 80 * 80, LAYER_42_SCALE, LAYER_42_ZP);

    // --- Layer 43 (/model.18/m.0/cv1/conv/Conv) ---
    // Input: 40x40x64 | Output Channels: 32 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_43_in, buf_43_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_43_pad_phy, buf_43_out_phy, wt_43_phy, bias_43_phy, 42, 42, 64, 32, 3, 1, LAYER_43_SCALE);
    cpu_silu(buf_43_out_virt, 40 * 40 * 32, LAYER_43_SCALE, LAYER_43_ZP);

    // --- Layer 44 (/model.18/m.0/cv2/conv/Conv) ---
    // Input: 40x40x32 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_44_in, buf_44_pad, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(buf_44_pad_phy, buf_44_out_phy, wt_44_phy, bias_44_phy, 42, 42, 32, 64, 3, 1, LAYER_44_SCALE);
    cpu_silu(buf_44_out_virt, 40 * 40 * 64, LAYER_44_SCALE, LAYER_44_ZP);

    // --- Layer 45 (/model.18/cv2/conv/Conv) ---
    // Input: 40x40x192 | Output Channels: 128 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_45_in, buf_45_pad, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(buf_45_pad_phy, buf_45_out_phy, wt_45_phy, bias_45_phy, 42, 42, 192, 128, 1, 1, LAYER_45_SCALE);
    cpu_silu(buf_45_out_virt, 40 * 40 * 128, LAYER_45_SCALE, LAYER_45_ZP);

    // --- Layer 46 (/model.19/conv/Conv) ---
    // Input: 40x40x128 | Output Channels: 128 | Kernel: 3x3 | Stride: 2
    cpu_pad(buf_46_in, buf_46_pad, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(buf_46_pad_phy, buf_46_out_phy, wt_46_phy, bias_46_phy, 42, 42, 128, 128, 3, 2, LAYER_46_SCALE);
    cpu_silu(buf_46_out_virt, 20 * 20 * 128, LAYER_46_SCALE, LAYER_46_ZP);

    // --- Layer 47 (/model.22/cv2.1/cv2.1.0/conv/Conv) ---
    // Input: 40x40x128 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_47_in, buf_47_pad, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(buf_47_pad_phy, buf_47_out_phy, wt_47_phy, bias_47_phy, 42, 42, 128, 64, 3, 1, LAYER_47_SCALE);
    cpu_silu(buf_47_out_virt, 40 * 40 * 64, LAYER_47_SCALE, LAYER_47_ZP);

    // --- Layer 48 (/model.22/cv3.1/cv3.1.0/conv/Conv) ---
    // Input: 40x40x128 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_48_in, buf_48_pad, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(buf_48_pad_phy, buf_48_out_phy, wt_48_phy, bias_48_phy, 42, 42, 128, 80, 3, 1, LAYER_48_SCALE);
    cpu_silu(buf_48_out_virt, 40 * 40 * 80, LAYER_48_SCALE, LAYER_48_ZP);

    // --- Layer 49 (/model.22/cv2.1/cv2.1.1/conv/Conv) ---
    // Input: 40x40x64 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_49_in, buf_49_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_49_pad_phy, buf_49_out_phy, wt_49_phy, bias_49_phy, 42, 42, 64, 64, 3, 1, LAYER_49_SCALE);
    cpu_silu(buf_49_out_virt, 40 * 40 * 64, LAYER_49_SCALE, LAYER_49_ZP);

    // --- Layer 50 (/model.22/cv3.1/cv3.1.1/conv/Conv) ---
    // Input: 40x40x80 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_50_in, buf_50_pad, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(buf_50_pad_phy, buf_50_out_phy, wt_50_phy, bias_50_phy, 42, 42, 80, 80, 3, 1, LAYER_50_SCALE);
    cpu_silu(buf_50_out_virt, 40 * 40 * 80, LAYER_50_SCALE, LAYER_50_ZP);

    // --- Layer 51 (/model.21/cv1/conv/Conv) ---
    // Input: 20x20x384 | Output Channels: 256 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_51_in, buf_51_pad, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(buf_51_pad_phy, buf_51_out_phy, wt_51_phy, bias_51_phy, 22, 22, 384, 256, 1, 1, LAYER_51_SCALE);
    cpu_silu(buf_51_out_virt, 20 * 20 * 256, LAYER_51_SCALE, LAYER_51_ZP);

    // --- Layer 52 (/model.22/cv2.1/cv2.1.2/Conv) ---
    // Input: 40x40x64 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_52_in, buf_52_pad, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(buf_52_pad_phy, buf_52_out_phy, wt_52_phy, bias_52_phy, 42, 42, 64, 64, 1, 1, LAYER_52_SCALE);
    cpu_silu(buf_52_out_virt, 40 * 40 * 64, LAYER_52_SCALE, LAYER_52_ZP);

    // --- Layer 53 (/model.22/cv3.1/cv3.1.2/Conv) ---
    // Input: 40x40x80 | Output Channels: 80 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_53_in, buf_53_pad, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(buf_53_pad_phy, buf_53_out_phy, wt_53_phy, bias_53_phy, 42, 42, 80, 80, 1, 1, LAYER_53_SCALE);
    cpu_silu(buf_53_out_virt, 40 * 40 * 80, LAYER_53_SCALE, LAYER_53_ZP);

    // --- Layer 54 (/model.21/m.0/cv1/conv/Conv) ---
    // Input: 20x20x128 | Output Channels: 72 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_54_in, buf_54_pad, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(buf_54_pad_phy, buf_54_out_phy, wt_54_phy, bias_54_phy, 22, 22, 128, 72, 3, 1, LAYER_54_SCALE);
    cpu_silu(buf_54_out_virt, 20 * 20 * 72, LAYER_54_SCALE, LAYER_54_ZP);

    // --- Layer 55 (/model.21/m.0/cv2/conv/Conv) ---
    // Input: 20x20x72 | Output Channels: 128 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_55_in, buf_55_pad, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(buf_55_pad_phy, buf_55_out_phy, wt_55_phy, bias_55_phy, 22, 22, 72, 128, 3, 1, LAYER_55_SCALE);
    cpu_silu(buf_55_out_virt, 20 * 20 * 128, LAYER_55_SCALE, LAYER_55_ZP);

    // --- Layer 56 (/model.21/cv2/conv/Conv) ---
    // Input: 20x20x384 | Output Channels: 256 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_56_in, buf_56_pad, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(buf_56_pad_phy, buf_56_out_phy, wt_56_phy, bias_56_phy, 22, 22, 384, 256, 1, 1, LAYER_56_SCALE);
    cpu_silu(buf_56_out_virt, 20 * 20 * 256, LAYER_56_SCALE, LAYER_56_ZP);

    // --- Layer 57 (/model.22/cv2.2/cv2.2.0/conv/Conv) ---
    // Input: 20x20x256 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_57_in, buf_57_pad, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(buf_57_pad_phy, buf_57_out_phy, wt_57_phy, bias_57_phy, 22, 22, 256, 64, 3, 1, LAYER_57_SCALE);
    cpu_silu(buf_57_out_virt, 20 * 20 * 64, LAYER_57_SCALE, LAYER_57_ZP);

    // --- Layer 58 (/model.22/cv3.2/cv3.2.0/conv/Conv) ---
    // Input: 20x20x256 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_58_in, buf_58_pad, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(buf_58_pad_phy, buf_58_out_phy, wt_58_phy, bias_58_phy, 22, 22, 256, 80, 3, 1, LAYER_58_SCALE);
    cpu_silu(buf_58_out_virt, 20 * 20 * 80, LAYER_58_SCALE, LAYER_58_ZP);

    // --- Layer 59 (/model.22/cv2.2/cv2.2.1/conv/Conv) ---
    // Input: 20x20x64 | Output Channels: 64 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_59_in, buf_59_pad, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(buf_59_pad_phy, buf_59_out_phy, wt_59_phy, bias_59_phy, 22, 22, 64, 64, 3, 1, LAYER_59_SCALE);
    cpu_silu(buf_59_out_virt, 20 * 20 * 64, LAYER_59_SCALE, LAYER_59_ZP);

    // --- Layer 60 (/model.22/cv3.2/cv3.2.1/conv/Conv) ---
    // Input: 20x20x80 | Output Channels: 80 | Kernel: 3x3 | Stride: 1
    cpu_pad(buf_60_in, buf_60_pad, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(buf_60_pad_phy, buf_60_out_phy, wt_60_phy, bias_60_phy, 22, 22, 80, 80, 3, 1, LAYER_60_SCALE);
    cpu_silu(buf_60_out_virt, 20 * 20 * 80, LAYER_60_SCALE, LAYER_60_ZP);

    // --- Layer 61 (/model.22/cv2.2/cv2.2.2/Conv) ---
    // Input: 20x20x64 | Output Channels: 64 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_61_in, buf_61_pad, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(buf_61_pad_phy, buf_61_out_phy, wt_61_phy, bias_61_phy, 22, 22, 64, 64, 1, 1, LAYER_61_SCALE);
    cpu_silu(buf_61_out_virt, 20 * 20 * 64, LAYER_61_SCALE, LAYER_61_ZP);

    // --- Layer 62 (/model.22/cv3.2/cv3.2.2/Conv) ---
    // Input: 20x20x80 | Output Channels: 80 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_62_in, buf_62_pad, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(buf_62_pad_phy, buf_62_out_phy, wt_62_phy, bias_62_phy, 22, 22, 80, 80, 1, 1, LAYER_62_SCALE);
    cpu_silu(buf_62_out_virt, 20 * 20 * 80, LAYER_62_SCALE, LAYER_62_ZP);

    // --- Layer 63 (/model.22/dfl/conv/Conv) ---
    // Input: 4x8400x16 | Output Channels: 1 | Kernel: 1x1 | Stride: 1
    cpu_pad(buf_63_in, buf_63_pad, 4, 8400, 16, INPUT_ZP);
    execute_fpga_layer(buf_63_pad_phy, buf_63_out_phy, wt_63_phy, bias_63_phy, 6, 8402, 16, 1, 1, 1, LAYER_63_SCALE);
    cpu_silu(buf_63_out_virt, 4 * 8400 * 1, LAYER_63_SCALE, LAYER_63_ZP);

}
