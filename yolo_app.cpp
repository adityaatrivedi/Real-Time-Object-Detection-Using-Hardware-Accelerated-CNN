#include <iostream>
#include <vector>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mmap.h>
#include <fstream>
#include <sstream>
#include <map>
#include <opencv2/opencv.hpp>

// =====================================================================
// HARDWARE REGISTERS & OFFSETS
// =====================================================================
#define DMA_BASE_ADDR  0xA0000000
#define YOLO_BASE_ADDR 0xA0010000

// DMA Offsets
#define MM2S_DMACR  (0x00 / 4)
#define MM2S_SA     (0x18 / 4)
#define MM2S_LENGTH (0x28 / 4)
#define S2MM_DMACR  (0x30 / 4)
#define S2MM_DA     (0x48 / 4)
#define S2MM_LENGTH (0x58 / 4)

// YOLO Top Offsets 
#define YOLO_CTRL   (0x00 / 4)
#define YOLO_WT_1   (0x10 / 4)
#define YOLO_BIAS_1 (0x1C / 4)
#define YOLO_H      (0x28 / 4)
#define YOLO_W      (0x30 / 4)
#define YOLO_IC     (0x38 / 4)
#define YOLO_OC     (0x40 / 4)
#define YOLO_K      (0x48 / 4)
#define YOLO_STRIDE (0x50 / 4)
#define YOLO_SCALE  (0x58 / 4)
#define YOLO_SHIFT  (0x60 / 4)

volatile unsigned int *dma_regs;
volatile unsigned int *yolo_regs;

// =====================================================================
// CMA (CONTIGUOUS MEMORY) MANAGEMENT VIA UIO
// =====================================================================
struct UioBuffer {
    int8_t* virt_addr;
    uint32_t phys_addr;
    size_t size;
};

UioBuffer cma_buffer;
std::vector<uint32_t> ordered_wt_phy;
std::vector<uint32_t> ordered_bias_phy;

// =====================================================================
// HELPER: FLOAT SCALE TO HARDWARE SCALE/SHIFT
// =====================================================================
void get_hw_scale_shift(float onnx_scale, int& hw_scale, int& hw_shift) {
    hw_shift = 16; 
    hw_scale = std::round(onnx_scale * (1 << hw_shift));
}

// =====================================================================
// 1. INIT_HARDWARE() & WEIGHT LOADER
// =====================================================================
bool init_hardware() {
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        std::cerr << "Failed to open /dev/mem. Need root." << std::endl;
        return false;
    }

    dma_regs = (volatile unsigned int *)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, DMA_BASE_ADDR);
    yolo_regs = (volatile unsigned int *)mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, YOLO_BASE_ADDR);

    int uio_fd = open("/dev/uio0", O_RDWR | O_SYNC);
    if (uio_fd < 0) {
        std::cerr << "Failed to open /dev/uio0. Check PetaLinux config." << std::endl;
        return false;
    }

    std::ifstream addr_file("/sys/class/uio/uio0/maps/map0/addr");
    std::string addr_str;
    std::getline(addr_file, addr_str);
    cma_buffer.phys_addr = std::stoul(addr_str, nullptr, 16);
    
    // Allocate 128MB chunk for all tensors and weights
    cma_buffer.size = 128 * 1024 * 1024; 
    cma_buffer.virt_addr = (int8_t*)mmap(NULL, cma_buffer.size, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd, 0);

    std::cout << "Hardware Initialized. AXI and CMA mapped." << std::endl;
    return true;
}

void load_weights(const std::string& map_file_path) {
    std::ifstream map_file(map_file_path);
    if (!map_file.is_open()) {
        std::cerr << "Failed to open " << map_file_path << std::endl;
        return;
    }

    std::string line;
    size_t weight_offset = 64 * 1024 * 1024; // Start loading at 64MB mark

    std::getline(map_file, line); // Skip header

    while (std::getline(map_file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string name, filename, size_str, dtype;
        
        std::getline(ss, name, ',');
        std::getline(ss, filename, ',');
        std::getline(ss, size_str, ',');
        std::getline(ss, dtype, ',');
        
        filename.erase(0, filename.find_first_not_of(" "));
        size_str.erase(0, size_str.find_first_not_of(" "));

        if (name.find("weight_quantized") != std::string::npos || name.find("bias_quantized") != std::string::npos) {
            int size_bytes = std::stoi(size_str);
            std::ifstream bin_file(filename, std::ios::binary);
            
            if (bin_file.is_open()) {
                bin_file.read((char*)(cma_buffer.virt_addr + weight_offset), size_bytes);
                uint32_t phy_addr = cma_buffer.phys_addr + weight_offset;
                
                if (name.find("weight") != std::string::npos) ordered_wt_phy.push_back(phy_addr);
                if (name.find("bias") != std::string::npos) ordered_bias_phy.push_back(phy_addr);
                
                weight_offset += size_bytes;
                if (weight_offset % 4096 != 0) weight_offset = (weight_offset / 4096 + 1) * 4096;
            } else {
                std::cerr << "Warning: Could not read " << filename << std::endl;
            }
        }
    }
    std::cout << "Loaded " << ordered_wt_phy.size() << " convolution layers into memory." << std::endl;
}

// =====================================================================
// 2. PREPROCESS()
// =====================================================================
void preprocess(const cv::Mat& original, int8_t* fpga_input, float input_scale, int8_t input_zp, 
                float& scale_factor, int& pad_w, int& pad_h) {
    int target_size = 640;
    scale_factor = std::min((float)target_size / original.cols, (float)target_size / original.rows);
    
    int new_w = std::round(original.cols * scale_factor);
    int new_h = std::round(original.rows * scale_factor);

    cv::Mat resized;
    cv::resize(original, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    pad_w = std::round((target_size - new_w) / 2.0f - 0.1f);
    pad_h = std::round((target_size - new_h) / 2.0f - 0.1f);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_h, target_size - new_h - pad_h, pad_w, target_size - new_w - pad_w, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < target_size; h++) {
            for (int w = 0; w < target_size; w++) {
                float val = padded.at<cv::Vec3b>(h, w)[c] / 255.0f;
                int q = std::round(val / input_scale) + input_zp;
                fpga_input[c * target_size * target_size + h * target_size + w] = (int8_t)std::clamp(q, -128, 127);
            }
        }
    }
}

// =====================================================================
// CPU TENSOR OPERATIONS
// =====================================================================
void cpu_pad(int8_t* in, int8_t* out, int H, int W, int C, int8_t zp) {
    int nH = H + 2, nW = W + 2;
    memset(out, zp, C * nH * nW);
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            memcpy(&out[c * nH * nW + (y + 1) * nW + 1], &in[c * H * W + y * W], W);
        }
    }
}

void cpu_silu(int8_t* buf, int size, float scale, int8_t zp) {
    for (int i = 0; i < size; i++) {
        float x = (buf[i] - zp) * scale;
        float silu = x / (1.0f + std::exp(-x));
        buf[i] = (int8_t)std::clamp((int)std::round((silu / scale) + zp), -128, 127);
    }
}

void cpu_upsample(int8_t* in, int8_t* out, int H, int W, int C, int scale) {
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H * scale; y++) {
            for (int x = 0; x < W * scale; x++) {
                out[c * (H * scale) * (W * scale) + y * (W * scale) + x] = in[c * H * W + (y / scale) * W + (x / scale)];
            }
        }
    }
}

void cpu_concat_append(int8_t* in_buf, int IC, int8_t* out_buf, int channel_offset, int H, int W) {
    int spatial_size = H * W;
    int8_t* dest_ptr = out_buf + (channel_offset * spatial_size);
    memcpy(dest_ptr, in_buf, IC * spatial_size * sizeof(int8_t));
}

void cpu_maxpool_5x5(int8_t* in, int8_t* out, int H, int W, int C) {
    int pad = 2; 
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int8_t max_val = -128;
                for (int ky = -pad; ky <= pad; ky++) {
                    for (int kx = -pad; kx <= pad; kx++) {
                        int py = y + ky;
                        int px = x + kx;
                        if (py >= 0 && py < H && px >= 0 && px < W) {
                            int8_t val = in[c * H * W + py * W + px];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                out[c * H * W + y * W + x] = max_val;
            }
        }
    }
}

// =====================================================================
// POST-PROCESSING DECODER (DFL HEAD)
// =====================================================================
void decode_yolo_branch(int8_t* bbox_int8, int8_t* cls_int8, float bbox_scale, float cls_scale,
                        int stride, int grid_size, std::vector<cv::Rect>& boxes, 
                        std::vector<float>& confs, std::vector<int>& class_ids,
                        int pad_w, int pad_h, float scale_factor) {
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            int spatial_idx = y * grid_size + x;

            // 1. Find max class confidence
            float max_conf = 0.0f;
            int best_cls = -1;
            for (int c = 0; c < 80; c++) {
                float val = cls_int8[c * grid_size * grid_size + spatial_idx] * cls_scale;
                float sigmoid = 1.0f / (1.0f + std::exp(-val));
                if (sigmoid > max_conf) {
                    max_conf = sigmoid;
                    best_cls = c;
                }
            }

            if (max_conf > 0.25f) {
                // 2. Decode DFL for 4 coordinates (Softmax)
                float dist[4];
                for (int i = 0; i < 4; i++) {
                    float sum = 0.0f, softmax_sum = 0.0f;
                    float exp_vals[16];

                    for (int j = 0; j < 16; j++) {
                        float val = bbox_int8[(i * 16 + j) * grid_size * grid_size + spatial_idx] * bbox_scale;
                        exp_vals[j] = std::exp(val);
                        softmax_sum += exp_vals[j];
                    }
                    for (int j = 0; j < 16; j++) {
                        sum += (exp_vals[j] / softmax_sum) * j;
                    }
                    dist[i] = sum;
                }

                // 3. Convert to Bounding Box
                float pb_cx = (x + 0.5f - dist[0] + x + 0.5f + dist[2]) / 2.0f * stride;
                float pb_cy = (y + 0.5f - dist[1] + y + 0.5f + dist[3]) / 2.0f * stride;
                float pb_w = (dist[0] + dist[2]) * stride;
                float pb_h = (dist[1] + dist[3]) * stride;

                int final_x = std::round((pb_cx - pb_w / 2.0f - pad_w) / scale_factor);
                int final_y = std::round((pb_cy - pb_h / 2.0f - pad_h) / scale_factor);
                int final_w = std::round(pb_w / scale_factor);
                int final_h = std::round(pb_h / scale_factor);

                boxes.push_back(cv::Rect(final_x, final_y, final_w, final_h));
                confs.push_back(max_conf);
                class_ids.push_back(best_cls);
            }
        }
    }
}

// =====================================================================
// 3. RUN_NETWORK()
// =====================================================================
void execute_fpga_layer(int8_t* src_virt, int8_t* dst_virt, uint32_t wt_phy, uint32_t bias_phy,
                        int H, int W, int IC, int OC, int K, int stride, float onnx_scale) {
    
    uint32_t src_phy = cma_buffer.phys_addr + (src_virt - cma_buffer.virt_addr);
    uint32_t dst_phy = cma_buffer.phys_addr + (dst_virt - cma_buffer.virt_addr);

    int hw_scale, hw_shift;
    get_hw_scale_shift(onnx_scale, hw_scale, hw_shift);

    yolo_regs[YOLO_WT_1]   = wt_phy;
    yolo_regs[YOLO_BIAS_1] = bias_phy;
    yolo_regs[YOLO_H]      = H;
    yolo_regs[YOLO_W]      = W;
    yolo_regs[YOLO_IC]     = IC;
    yolo_regs[YOLO_OC]     = OC;
    yolo_regs[YOLO_K]      = K;
    yolo_regs[YOLO_STRIDE] = stride;
    yolo_regs[YOLO_SCALE]  = hw_scale;
    yolo_regs[YOLO_SHIFT]  = hw_shift;

    yolo_regs[YOLO_CTRL] = 0x1;

    int out_h = ((H - K) / stride) + 1;
    int out_w = ((W - K) / stride) + 1;
    
    dma_regs[MM2S_DMACR] = 0x4; 
    dma_regs[S2MM_DMACR] = 0x4; 
    usleep(100);
    dma_regs[MM2S_DMACR] = 0x1; 
    dma_regs[S2MM_DMACR] = 0x1; 

    dma_regs[S2MM_DA]     = dst_phy; 
    dma_regs[S2MM_LENGTH] = out_h * out_w * OC; 

    dma_regs[MM2S_SA]     = src_phy; 
    dma_regs[MM2S_LENGTH] = H * W * IC; 

    while (!(dma_regs[0x34 / 4] & 0x2)); 
    while (!(yolo_regs[YOLO_CTRL] & 0x2)); 
}

// --- AUTO-GENERATED YOLOv8 SCHEDULER ---
void run_network(int8_t* cma_base_addr, float scale_factor, int pad_w, int pad_h, cv::Mat& img) {
    size_t current_offset = 0;
    
    uint32_t wt_phy[1000] = {0};   
    uint32_t bias_phy[1000] = {0}; 
    float layer_scales[1000] = {0};
    int8_t layer_zps[1000] = {0};
    int8_t INPUT_ZP = 0;

    // Automap physical memory addresses to node IDs
    int node_ids[] = {129, 136, 143, 151, 158, 167, 174, 181, 189, 196, 204, 211, 220, 227, 234, 242, 249, 257, 264, 273, 280, 287, 295, 302, 311, 318, 329, 338, 346, 353, 361, 370, 378, 385, 393, 400, 401, 402, 422, 423, 424, 443, 444, 448, 459, 467, 474, 475, 476, 496, 497, 498, 517, 518, 522, 533, 541, 548, 549, 562, 563, 576, 577, 591};
    for(size_t i = 0; i < ordered_wt_phy.size() && i < 64; i++) {
        wt_phy[node_ids[i]] = ordered_wt_phy[i];
        bias_phy[node_ids[i]] = ordered_bias_phy[i];
    }
    
    // PASTE the contents of c++_scales.txt right here!

        // --- AUTO-EXTRACTED SCALES AND ZERO POINTS ---
    layer_scales[0] = 0.5066584348678589f;
    layer_zps[0] = 126;
    layer_scales[1] = 1.341862440109253f;
    layer_zps[1] = 133;
    layer_scales[2] = 0.6876779794692993f;
    layer_zps[2] = 193;
    layer_scales[3] = 0.257639616727829f;
    layer_zps[3] = 117;
    layer_scales[4] = 0.3365197777748108f;
    layer_zps[4] = 154;
    layer_scales[5] = 0.2165534496307373f;
    layer_zps[5] = 156;
    layer_scales[6] = 0.11519859731197357f;
    layer_zps[6] = 163;
    layer_scales[7] = 0.10638369619846344f;
    layer_zps[7] = 160;
    layer_scales[8] = 0.056630946695804596f;
    layer_zps[8] = 154;
    layer_scales[9] = 0.06859970092773438f;
    layer_zps[9] = 140;
    layer_scales[10] = 0.046556778252124786f;
    layer_zps[10] = 155;
    layer_scales[11] = 0.09289347380399704f;
    layer_zps[11] = 143;
    layer_scales[12] = 0.08338603377342224f;
    layer_zps[12] = 149;
    layer_scales[13] = 0.09364522248506546f;
    layer_zps[13] = 161;
    layer_scales[14] = 0.08999158442020416f;
    layer_zps[14] = 153;
    layer_scales[15] = 0.06389555335044861f;
    layer_zps[15] = 140;
    layer_scales[16] = 0.07042430341243744f;
    layer_zps[16] = 154;
    layer_scales[17] = 0.055990707129240036f;
    layer_zps[17] = 137;
    layer_scales[18] = 0.11702887713909149f;
    layer_zps[18] = 135;
    layer_scales[19] = 0.07115522027015686f;
    layer_zps[19] = 140;
    layer_scales[20] = 0.08022060245275497f;
    layer_zps[20] = 128;
    layer_scales[21] = 0.12202352285385132f;
    layer_zps[21] = 147;
    layer_scales[22] = 0.10731935501098633f;
    layer_zps[22] = 105;
    layer_scales[23] = 0.15397877991199493f;
    layer_zps[23] = 126;
    layer_scales[24] = 0.10150192677974701f;
    layer_zps[24] = 150;
    layer_scales[25] = 0.0925111398100853f;
    layer_zps[25] = 159;
    layer_scales[26] = 0.06890851259231567f;
    layer_zps[26] = 153;
    layer_scales[27] = 0.07348445802927017f;
    layer_zps[27] = 136;
    layer_scales[28] = 0.05500994622707367f;
    layer_zps[28] = 152;
    layer_scales[29] = 0.07505283504724503f;
    layer_zps[29] = 132;
    layer_scales[30] = 0.08933011442422867f;
    layer_zps[30] = 132;
    layer_scales[31] = 0.06389693915843964f;
    layer_zps[31] = 152;
    layer_scales[32] = 0.04251056909561157f;
    layer_zps[32] = 146;
    layer_scales[33] = 0.0705355554819107f;
    layer_zps[33] = 140;
    layer_scales[34] = 0.0664665549993515f;
    layer_zps[34] = 160;
    layer_scales[35] = 0.07513915747404099f;
    layer_zps[35] = 129;
    layer_scales[36] = 0.11574936658143997f;
    layer_zps[36] = 185;
    layer_scales[37] = 0.08539995551109314f;
    layer_zps[37] = 175;
    layer_scales[38] = 0.34859099984169006f;
    layer_zps[38] = 126;
    layer_scales[39] = 0.739556074142456f;
    layer_zps[39] = 158;
    layer_scales[40] = 0.09971887618303299f;
    layer_zps[40] = 148;
    layer_scales[41] = 0.14025498926639557f;
    layer_zps[41] = 64;
    layer_scales[42] = 0.13142980635166168f;
    layer_zps[42] = 248;
    layer_scales[43] = 0.058265283703804016f;
    layer_zps[43] = 157;
    layer_scales[44] = 0.12419674545526505f;
    layer_zps[44] = 141;
    layer_scales[45] = 0.12456105649471283f;
    layer_zps[45] = 182;
    layer_scales[46] = 0.09176313132047653f;
    layer_zps[46] = 153;
    layer_scales[47] = 0.16017718613147736f;
    layer_zps[47] = 139;
    layer_scales[48] = 0.09500206261873245f;
    layer_zps[48] = 147;
    layer_scales[49] = 0.3458927571773529f;
    layer_zps[49] = 91;
    layer_scales[50] = 0.9854210019111633f;
    layer_zps[50] = 119;
    layer_scales[51] = 0.09624502062797546f;
    layer_zps[51] = 131;
    layer_scales[52] = 0.10997049510478973f;
    layer_zps[52] = 71;
    layer_scales[53] = 0.22314687073230743f;
    layer_zps[53] = 247;
    layer_scales[54] = 0.06883151829242706f;
    layer_zps[54] = 142;
    layer_scales[55] = 0.11994030326604843f;
    layer_zps[55] = 136;
    layer_scales[56] = 0.16480693221092224f;
    layer_zps[56] = 182;
    layer_scales[57] = 0.14965465664863586f;
    layer_zps[57] = 161;
    layer_scales[58] = 0.1689160019159317f;
    layer_zps[58] = 186;
    layer_scales[59] = 0.2625857889652252f;
    layer_zps[59] = 122;
    layer_scales[60] = 0.8066868185997009f;
    layer_zps[60] = 133;
    layer_scales[61] = 0.09514147788286209f;
    layer_zps[61] = 76;
    layer_scales[62] = 0.23956137895584106f;
    layer_zps[62] = 241;
    layer_scales[63] = 0.05595121532678604f;
    layer_zps[63] = 0;




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
    execute_fpga_layer(pad_129, buf__model_0_conv_Conv_output_0, wt_phy[129], bias_phy[129], 642, 642, 3, 16, 3, 2, layer_scales[129]);
    cpu_silu(buf__model_0_conv_Conv_output_0, 1638400, layer_scales[129], layer_zps[129]);

    // --- Node 136: Conv ---
    int8_t* buf__model_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(1638400);
    int8_t* buf__model_1_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_136 = alloc_buf(1658944);
    cpu_pad(buf__model_0_act_Mul_output_0_DequantizeLinear_Output, pad_136, 320, 320, 16, INPUT_ZP);
    execute_fpga_layer(pad_136, buf__model_1_conv_Conv_output_0, wt_phy[136], bias_phy[136], 322, 322, 16, 32, 3, 2, layer_scales[136]);
    cpu_silu(buf__model_1_conv_Conv_output_0, 819200, layer_scales[136], layer_zps[136]);

    // --- Node 143: Conv ---
    int8_t* buf__model_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(819200);
    int8_t* buf__model_2_cv1_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_143 = alloc_buf(839808);
    cpu_pad(buf__model_1_act_Mul_output_0_DequantizeLinear_Output, pad_143, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(pad_143, buf__model_2_cv1_conv_Conv_output_0, wt_phy[143], bias_phy[143], 162, 162, 32, 32, 3, 1, layer_scales[143]);
    cpu_silu(buf__model_2_cv1_conv_Conv_output_0, 819200, layer_scales[143], layer_zps[143]);

    // --- Node 151: Conv ---
    int8_t* buf__model_2_Split_output_1_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_2_m_0_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_151 = alloc_buf(419904);
    cpu_pad(buf__model_2_Split_output_1_DequantizeLinear_Output, pad_151, 160, 160, 16, INPUT_ZP);
    execute_fpga_layer(pad_151, buf__model_2_m_0_cv1_conv_Conv_output_0, wt_phy[151], bias_phy[151], 162, 162, 16, 8, 3, 1, layer_scales[151]);
    cpu_silu(buf__model_2_m_0_cv1_conv_Conv_output_0, 204800, layer_scales[151], layer_zps[151]);

    // --- Node 158: Conv ---
    int8_t* buf__model_2_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_2_m_0_cv2_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_158 = alloc_buf(209952);
    cpu_pad(buf__model_2_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_158, 160, 160, 8, INPUT_ZP);
    execute_fpga_layer(pad_158, buf__model_2_m_0_cv2_conv_Conv_output_0, wt_phy[158], bias_phy[158], 162, 162, 8, 16, 3, 1, layer_scales[158]);
    cpu_silu(buf__model_2_m_0_cv2_conv_Conv_output_0, 409600, layer_scales[158], layer_zps[158]);

    // --- Node 164: Concat ---
    int8_t* buf__model_2_Split_output_0 = alloc_buf(409600); 
    int8_t* buf__model_2_m_0_Add_output_0 = alloc_buf(409600);
    int8_t* buf__model_2_Concat_output_0 = alloc_buf(1228800);
    cpu_concat_append(buf__model_2_Split_output_0, 16, buf__model_2_Concat_output_0, 0, 160, 160);
    cpu_concat_append(buf__model_2_Split_output_1_DequantizeLinear_Output, 16, buf__model_2_Concat_output_0, 16, 160, 160);
    cpu_concat_append(buf__model_2_m_0_Add_output_0, 16, buf__model_2_Concat_output_0, 32, 160, 160);

    // --- Node 167: Conv ---
    int8_t* buf__model_2_Concat_output_0_DequantizeLinear_Output = alloc_buf(1228800);
    int8_t* buf__model_2_cv2_conv_Conv_output_0 = alloc_buf(819200);
    int8_t* pad_167 = alloc_buf(1259712);
    cpu_pad(buf__model_2_Concat_output_0_DequantizeLinear_Output, pad_167, 160, 160, 48, INPUT_ZP);
    execute_fpga_layer(pad_167, buf__model_2_cv2_conv_Conv_output_0, wt_phy[167], bias_phy[167], 162, 162, 48, 32, 3, 1, layer_scales[167]);
    cpu_silu(buf__model_2_cv2_conv_Conv_output_0, 819200, layer_scales[167], layer_zps[167]);

    // --- Node 174: Conv ---
    int8_t* buf__model_2_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(819200);
    int8_t* buf__model_3_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_174 = alloc_buf(839808);
    cpu_pad(buf__model_2_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_174, 160, 160, 32, INPUT_ZP);
    execute_fpga_layer(pad_174, buf__model_3_conv_Conv_output_0, wt_phy[174], bias_phy[174], 162, 162, 32, 64, 3, 2, layer_scales[174]);
    cpu_silu(buf__model_3_conv_Conv_output_0, 409600, layer_scales[174], layer_zps[174]);

    // --- Node 181: Conv ---
    int8_t* buf__model_3_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_4_cv1_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_181 = alloc_buf(430336);
    cpu_pad(buf__model_3_act_Mul_output_0_DequantizeLinear_Output, pad_181, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_181, buf__model_4_cv1_conv_Conv_output_0, wt_phy[181], bias_phy[181], 82, 82, 64, 64, 3, 1, layer_scales[181]);
    cpu_silu(buf__model_4_cv1_conv_Conv_output_0, 409600, layer_scales[181], layer_zps[181]);

    // --- Node 189: Conv ---
    int8_t* buf__model_4_Split_output_1_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_4_m_0_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_189 = alloc_buf(215168);
    cpu_pad(buf__model_4_Split_output_1_DequantizeLinear_Output, pad_189, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_189, buf__model_4_m_0_cv1_conv_Conv_output_0, wt_phy[189], bias_phy[189], 82, 82, 32, 16, 3, 1, layer_scales[189]);
    cpu_silu(buf__model_4_m_0_cv1_conv_Conv_output_0, 102400, layer_scales[189], layer_zps[189]);

    // --- Node 196: Conv ---
    int8_t* buf__model_4_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_4_m_0_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_196 = alloc_buf(107584);
    cpu_pad(buf__model_4_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_196, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_196, buf__model_4_m_0_cv2_conv_Conv_output_0, wt_phy[196], bias_phy[196], 82, 82, 16, 32, 3, 1, layer_scales[196]);
    cpu_silu(buf__model_4_m_0_cv2_conv_Conv_output_0, 204800, layer_scales[196], layer_zps[196]);

    // --- Node 204: Conv ---
    int8_t* buf__model_4_m_0_Add_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_4_m_1_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_204 = alloc_buf(215168);
    cpu_pad(buf__model_4_m_0_Add_output_0_DequantizeLinear_Output, pad_204, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_204, buf__model_4_m_1_cv1_conv_Conv_output_0, wt_phy[204], bias_phy[204], 82, 82, 32, 16, 3, 1, layer_scales[204]);
    cpu_silu(buf__model_4_m_1_cv1_conv_Conv_output_0, 102400, layer_scales[204], layer_zps[204]);

    // --- Node 211: Conv ---
    int8_t* buf__model_4_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_4_m_1_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_211 = alloc_buf(107584);
    cpu_pad(buf__model_4_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_211, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_211, buf__model_4_m_1_cv2_conv_Conv_output_0, wt_phy[211], bias_phy[211], 82, 82, 16, 32, 3, 1, layer_scales[211]);
    cpu_silu(buf__model_4_m_1_cv2_conv_Conv_output_0, 204800, layer_scales[211], layer_zps[211]);

    // --- Node 217: Concat ---
    int8_t* buf__model_4_Split_output_0 = alloc_buf(204800);
    int8_t* buf__model_4_m_1_Add_output_0 = alloc_buf(204800);
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
    execute_fpga_layer(pad_220, buf__model_4_cv2_conv_Conv_output_0, wt_phy[220], bias_phy[220], 82, 82, 128, 64, 3, 1, layer_scales[220]);
    cpu_silu(buf__model_4_cv2_conv_Conv_output_0, 409600, layer_scales[220], layer_zps[220]);

    // --- Node 227: Conv ---
    int8_t* buf__model_4_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_5_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_227 = alloc_buf(430336);
    cpu_pad(buf__model_4_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_227, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_227, buf__model_5_conv_Conv_output_0, wt_phy[227], bias_phy[227], 82, 82, 64, 128, 3, 2, layer_scales[227]);
    cpu_silu(buf__model_5_conv_Conv_output_0, 204800, layer_scales[227], layer_zps[227]);

    // --- Node 234: Conv ---
    int8_t* buf__model_5_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_6_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_234 = alloc_buf(225792);
    cpu_pad(buf__model_5_act_Mul_output_0_DequantizeLinear_Output, pad_234, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_234, buf__model_6_cv1_conv_Conv_output_0, wt_phy[234], bias_phy[234], 42, 42, 128, 128, 3, 1, layer_scales[234]);
    cpu_silu(buf__model_6_cv1_conv_Conv_output_0, 204800, layer_scales[234], layer_zps[234]);

    // --- Node 242: Conv ---
    int8_t* buf__model_6_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_6_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_242 = alloc_buf(112896);
    cpu_pad(buf__model_6_Split_output_1_DequantizeLinear_Output, pad_242, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_242, buf__model_6_m_0_cv1_conv_Conv_output_0, wt_phy[242], bias_phy[242], 42, 42, 64, 32, 3, 1, layer_scales[242]);
    cpu_silu(buf__model_6_m_0_cv1_conv_Conv_output_0, 51200, layer_scales[242], layer_zps[242]);

    // --- Node 249: Conv ---
    int8_t* buf__model_6_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_6_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_249 = alloc_buf(56448);
    cpu_pad(buf__model_6_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_249, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_249, buf__model_6_m_0_cv2_conv_Conv_output_0, wt_phy[249], bias_phy[249], 42, 42, 32, 64, 3, 1, layer_scales[249]);
    cpu_silu(buf__model_6_m_0_cv2_conv_Conv_output_0, 102400, layer_scales[249], layer_zps[249]);

    // --- Node 257: Conv ---
    int8_t* buf__model_6_m_0_Add_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_6_m_1_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_257 = alloc_buf(112896);
    cpu_pad(buf__model_6_m_0_Add_output_0_DequantizeLinear_Output, pad_257, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_257, buf__model_6_m_1_cv1_conv_Conv_output_0, wt_phy[257], bias_phy[257], 42, 42, 64, 32, 3, 1, layer_scales[257]);
    cpu_silu(buf__model_6_m_1_cv1_conv_Conv_output_0, 51200, layer_scales[257], layer_zps[257]);

    // --- Node 264: Conv ---
    int8_t* buf__model_6_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_6_m_1_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_264 = alloc_buf(56448);
    cpu_pad(buf__model_6_m_1_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_264, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_264, buf__model_6_m_1_cv2_conv_Conv_output_0, wt_phy[264], bias_phy[264], 42, 42, 32, 64, 3, 1, layer_scales[264]);
    cpu_silu(buf__model_6_m_1_cv2_conv_Conv_output_0, 102400, layer_scales[264], layer_zps[264]);

    // --- Node 270: Concat ---
    int8_t* buf__model_6_Split_output_0 = alloc_buf(102400);
    int8_t* buf__model_6_m_1_Add_output_0 = alloc_buf(102400);
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
    execute_fpga_layer(pad_273, buf__model_6_cv2_conv_Conv_output_0, wt_phy[273], bias_phy[273], 42, 42, 256, 128, 3, 1, layer_scales[273]);
    cpu_silu(buf__model_6_cv2_conv_Conv_output_0, 204800, layer_scales[273], layer_zps[273]);

    // --- Node 280: Conv ---
    int8_t* buf__model_6_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_7_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_280 = alloc_buf(225792);
    cpu_pad(buf__model_6_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_280, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_280, buf__model_7_conv_Conv_output_0, wt_phy[280], bias_phy[280], 42, 42, 128, 256, 3, 2, layer_scales[280]);
    cpu_silu(buf__model_7_conv_Conv_output_0, 102400, layer_scales[280], layer_zps[280]);

    // --- Node 287: Conv ---
    int8_t* buf__model_7_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_8_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_287 = alloc_buf(123904);
    cpu_pad(buf__model_7_act_Mul_output_0_DequantizeLinear_Output, pad_287, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_287, buf__model_8_cv1_conv_Conv_output_0, wt_phy[287], bias_phy[287], 22, 22, 256, 256, 3, 1, layer_scales[287]);
    cpu_silu(buf__model_8_cv1_conv_Conv_output_0, 102400, layer_scales[287], layer_zps[287]);

    // --- Node 295: Conv ---
    int8_t* buf__model_8_Split_output_1_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_8_m_0_cv1_conv_Conv_output_0 = alloc_buf(28800);
    int8_t* pad_295 = alloc_buf(61952);
    cpu_pad(buf__model_8_Split_output_1_DequantizeLinear_Output, pad_295, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(pad_295, buf__model_8_m_0_cv1_conv_Conv_output_0, wt_phy[295], bias_phy[295], 22, 22, 128, 72, 3, 1, layer_scales[295]);
    cpu_silu(buf__model_8_m_0_cv1_conv_Conv_output_0, 28800, layer_scales[295], layer_zps[295]);

    // --- Node 302: Conv ---
    int8_t* buf__model_8_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(28800);
    int8_t* buf__model_8_m_0_cv2_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_302 = alloc_buf(34848);
    cpu_pad(buf__model_8_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_302, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(pad_302, buf__model_8_m_0_cv2_conv_Conv_output_0, wt_phy[302], bias_phy[302], 22, 22, 72, 128, 3, 1, layer_scales[302]);
    cpu_silu(buf__model_8_m_0_cv2_conv_Conv_output_0, 51200, layer_scales[302], layer_zps[302]);

    // --- Node 308: Concat ---
    int8_t* buf__model_8_Split_output_0 = alloc_buf(51200);
    int8_t* buf__model_8_m_0_Add_output_0 = alloc_buf(51200);
    int8_t* buf__model_8_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_8_Split_output_0, 128, buf__model_8_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_8_Split_output_1_DequantizeLinear_Output, 128, buf__model_8_Concat_output_0, 128, 20, 20);
    cpu_concat_append(buf__model_8_m_0_Add_output_0, 128, buf__model_8_Concat_output_0, 256, 20, 20);

    // --- Node 311: Conv ---
    int8_t* buf__model_8_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_8_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_311 = alloc_buf(185856);
    cpu_pad(buf__model_8_Concat_output_0_DequantizeLinear_Output, pad_311, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_311, buf__model_8_cv2_conv_Conv_output_0, wt_phy[311], bias_phy[311], 22, 22, 384, 256, 3, 1, layer_scales[311]);
    cpu_silu(buf__model_8_cv2_conv_Conv_output_0, 102400, layer_scales[311], layer_zps[311]);

    // --- Node 318: Conv ---
    int8_t* buf__model_8_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_9_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_318 = alloc_buf(123904);
    cpu_pad(buf__model_8_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_318, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_318, buf__model_9_cv1_conv_Conv_output_0, wt_phy[318], bias_phy[318], 22, 22, 256, 128, 3, 1, layer_scales[318]);
    cpu_silu(buf__model_9_cv1_conv_Conv_output_0, 51200, layer_scales[318], layer_zps[318]);

    // --- Node 323: MaxPool ---
    int8_t* buf__model_9_cv1_act_Mul_output_0 = alloc_buf(51200);
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
    execute_fpga_layer(pad_329, buf__model_9_cv2_conv_Conv_output_0, wt_phy[329], bias_phy[329], 22, 22, 512, 256, 3, 1, layer_scales[329]);
    cpu_silu(buf__model_9_cv2_conv_Conv_output_0, 102400, layer_scales[329], layer_zps[329]);

    // --- Node 334: Upsample ---
    int8_t* buf__model_9_cv2_act_Mul_output_0 = alloc_buf(102400);
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
    execute_fpga_layer(pad_338, buf__model_12_cv1_conv_Conv_output_0, wt_phy[338], bias_phy[338], 42, 42, 384, 128, 3, 1, layer_scales[338]);
    cpu_silu(buf__model_12_cv1_conv_Conv_output_0, 204800, layer_scales[338], layer_zps[338]);

    // --- Node 346: Conv ---
    int8_t* buf__model_12_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_12_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_346 = alloc_buf(112896);
    cpu_pad(buf__model_12_Split_output_1_DequantizeLinear_Output, pad_346, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_346, buf__model_12_m_0_cv1_conv_Conv_output_0, wt_phy[346], bias_phy[346], 42, 42, 64, 32, 3, 1, layer_scales[346]);
    cpu_silu(buf__model_12_m_0_cv1_conv_Conv_output_0, 51200, layer_scales[346], layer_zps[346]);

    // --- Node 353: Conv ---
    int8_t* buf__model_12_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_12_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_353 = alloc_buf(56448);
    cpu_pad(buf__model_12_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_353, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_353, buf__model_12_m_0_cv2_conv_Conv_output_0, wt_phy[353], bias_phy[353], 42, 42, 32, 64, 3, 1, layer_scales[353]);
    cpu_silu(buf__model_12_m_0_cv2_conv_Conv_output_0, 102400, layer_scales[353], layer_zps[353]);

    // --- Node 358: Concat ---
    int8_t* buf__model_12_Split_output_0 = alloc_buf(102400);
    int8_t* buf__model_12_m_0_cv2_act_Mul_output_0 = alloc_buf(102400);
    int8_t* buf__model_12_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_12_Split_output_0, 64, buf__model_12_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_12_Split_output_1_DequantizeLinear_Output, 64, buf__model_12_Concat_output_0, 64, 40, 40);
    cpu_concat_append(buf__model_12_m_0_cv2_act_Mul_output_0, 64, buf__model_12_Concat_output_0, 128, 40, 40);

    // --- Node 361: Conv ---
    int8_t* buf__model_12_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_12_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_361 = alloc_buf(338688);
    cpu_pad(buf__model_12_Concat_output_0_DequantizeLinear_Output, pad_361, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_361, buf__model_12_cv2_conv_Conv_output_0, wt_phy[361], bias_phy[361], 42, 42, 192, 128, 3, 1, layer_scales[361]);
    cpu_silu(buf__model_12_cv2_conv_Conv_output_0, 204800, layer_scales[361], layer_zps[361]);

    // --- Node 366: Upsample ---
    int8_t* buf__model_12_cv2_act_Mul_output_0 = alloc_buf(204800);
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
    execute_fpga_layer(pad_370, buf__model_15_cv1_conv_Conv_output_0, wt_phy[370], bias_phy[370], 82, 82, 192, 64, 3, 1, layer_scales[370]);
    cpu_silu(buf__model_15_cv1_conv_Conv_output_0, 409600, layer_scales[370], layer_zps[370]);

    // --- Node 378: Conv ---
    int8_t* buf__model_15_Split_output_1_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_15_m_0_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_378 = alloc_buf(215168);
    cpu_pad(buf__model_15_Split_output_1_DequantizeLinear_Output, pad_378, 80, 80, 32, INPUT_ZP);
    execute_fpga_layer(pad_378, buf__model_15_m_0_cv1_conv_Conv_output_0, wt_phy[378], bias_phy[378], 82, 82, 32, 16, 3, 1, layer_scales[378]);
    cpu_silu(buf__model_15_m_0_cv1_conv_Conv_output_0, 102400, layer_scales[378], layer_zps[378]);

    // --- Node 385: Conv ---
    int8_t* buf__model_15_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_15_m_0_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_385 = alloc_buf(107584);
    cpu_pad(buf__model_15_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_385, 80, 80, 16, INPUT_ZP);
    execute_fpga_layer(pad_385, buf__model_15_m_0_cv2_conv_Conv_output_0, wt_phy[385], bias_phy[385], 82, 82, 16, 32, 3, 1, layer_scales[385]);
    cpu_silu(buf__model_15_m_0_cv2_conv_Conv_output_0, 204800, layer_scales[385], layer_zps[385]);

    // --- Node 390: Concat ---
    int8_t* buf__model_15_Split_output_0 = alloc_buf(204800);
    int8_t* buf__model_15_m_0_cv2_act_Mul_output_0 = alloc_buf(204800);
    int8_t* buf__model_15_Concat_output_0 = alloc_buf(614400);
    cpu_concat_append(buf__model_15_Split_output_0, 32, buf__model_15_Concat_output_0, 0, 80, 80);
    cpu_concat_append(buf__model_15_Split_output_1_DequantizeLinear_Output, 32, buf__model_15_Concat_output_0, 32, 80, 80);
    cpu_concat_append(buf__model_15_m_0_cv2_act_Mul_output_0, 32, buf__model_15_Concat_output_0, 64, 80, 80);

    // --- Node 393: Conv ---
    int8_t* buf__model_15_Concat_output_0_DequantizeLinear_Output = alloc_buf(614400);
    int8_t* buf__model_15_cv2_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_393 = alloc_buf(645504);
    cpu_pad(buf__model_15_Concat_output_0_DequantizeLinear_Output, pad_393, 80, 80, 96, INPUT_ZP);
    execute_fpga_layer(pad_393, buf__model_15_cv2_conv_Conv_output_0, wt_phy[393], bias_phy[393], 82, 82, 96, 64, 3, 1, layer_scales[393]);
    cpu_silu(buf__model_15_cv2_conv_Conv_output_0, 409600, layer_scales[393], layer_zps[393]);

    // --- Node 400: Conv ---
    int8_t* buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_16_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_400 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_400, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_400, buf__model_16_conv_Conv_output_0, wt_phy[400], bias_phy[400], 82, 82, 64, 64, 3, 2, layer_scales[400]);
    cpu_silu(buf__model_16_conv_Conv_output_0, 102400, layer_scales[400], layer_zps[400]);

    // --- Node 401: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_401 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_401, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_401, buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0, wt_phy[401], bias_phy[401], 82, 82, 64, 64, 3, 1, layer_scales[401]);
    cpu_silu(buf__model_22_cv2_0_cv2_0_0_conv_Conv_output_0, 409600, layer_scales[401], layer_zps[401]);

    // --- Node 402: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_402 = alloc_buf(430336);
    cpu_pad(buf__model_15_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_402, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_402, buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0, wt_phy[402], bias_phy[402], 82, 82, 64, 80, 3, 1, layer_scales[402]);
    cpu_silu(buf__model_22_cv3_0_cv3_0_0_conv_Conv_output_0, 512000, layer_scales[402], layer_zps[402]);

    // --- Node 415: Concat ---
    int8_t* buf__model_16_act_Mul_output_0 = alloc_buf(102400);
    int8_t* buf__model_17_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_16_act_Mul_output_0, 64, buf__model_17_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_12_cv2_act_Mul_output_0, 128, buf__model_17_Concat_output_0, 64, 40, 40);

    // --- Node 422: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_422 = alloc_buf(430336);
    cpu_pad(buf__model_22_cv2_0_cv2_0_0_act_Mul_output_0_DequantizeLinear_Output, pad_422, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_422, buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0, wt_phy[422], bias_phy[422], 82, 82, 64, 64, 3, 1, layer_scales[422]);
    cpu_silu(buf__model_22_cv2_0_cv2_0_1_conv_Conv_output_0, 409600, layer_scales[422], layer_zps[422]);

    // --- Node 423: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(512000);
    int8_t* buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_423 = alloc_buf(537920);
    cpu_pad(buf__model_22_cv3_0_cv3_0_0_act_Mul_output_0_DequantizeLinear_Output, pad_423, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(pad_423, buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0, wt_phy[423], bias_phy[423], 82, 82, 80, 80, 3, 1, layer_scales[423]);
    cpu_silu(buf__model_22_cv3_0_cv3_0_1_conv_Conv_output_0, 512000, layer_scales[423], layer_zps[423]);

    // --- Node 424: Conv ---
    int8_t* buf__model_17_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_18_cv1_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_424 = alloc_buf(338688);
    cpu_pad(buf__model_17_Concat_output_0_DequantizeLinear_Output, pad_424, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_424, buf__model_18_cv1_conv_Conv_output_0, wt_phy[424], bias_phy[424], 42, 42, 192, 128, 3, 1, layer_scales[424]);
    cpu_silu(buf__model_18_cv1_conv_Conv_output_0, 204800, layer_scales[424], layer_zps[424]);

    // --- Node 443: Conv ---
    int8_t* buf__model_22_cv2_0_cv2_0_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(409600);
    int8_t* buf__model_22_cv2_0_cv2_0_2_Conv_output_0 = alloc_buf(409600);
    int8_t* pad_443 = alloc_buf(430336);
    cpu_pad(buf__model_22_cv2_0_cv2_0_1_act_Mul_output_0_DequantizeLinear_Output, pad_443, 80, 80, 64, INPUT_ZP);
    execute_fpga_layer(pad_443, buf__model_22_cv2_0_cv2_0_2_Conv_output_0, wt_phy[443], bias_phy[443], 82, 82, 64, 64, 3, 1, layer_scales[443]);
    cpu_silu(buf__model_22_cv2_0_cv2_0_2_Conv_output_0, 409600, layer_scales[443], layer_zps[443]);

    // --- Node 444: Conv ---
    int8_t* buf__model_22_cv3_0_cv3_0_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(512000);
    int8_t* buf__model_22_cv3_0_cv3_0_2_Conv_output_0 = alloc_buf(512000);
    int8_t* pad_444 = alloc_buf(537920);
    cpu_pad(buf__model_22_cv3_0_cv3_0_1_act_Mul_output_0_DequantizeLinear_Output, pad_444, 80, 80, 80, INPUT_ZP);
    execute_fpga_layer(pad_444, buf__model_22_cv3_0_cv3_0_2_Conv_output_0, wt_phy[444], bias_phy[444], 82, 82, 80, 80, 3, 1, layer_scales[444]);
    cpu_silu(buf__model_22_cv3_0_cv3_0_2_Conv_output_0, 512000, layer_scales[444], layer_zps[444]);

    // --- Node 448: Conv ---
    int8_t* buf__model_18_Split_output_1_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_18_m_0_cv1_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_448 = alloc_buf(112896);
    cpu_pad(buf__model_18_Split_output_1_DequantizeLinear_Output, pad_448, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_448, buf__model_18_m_0_cv1_conv_Conv_output_0, wt_phy[448], bias_phy[448], 42, 42, 64, 32, 3, 1, layer_scales[448]);
    cpu_silu(buf__model_18_m_0_cv1_conv_Conv_output_0, 51200, layer_scales[448], layer_zps[448]);

    // --- Node 459: Conv ---
    int8_t* buf__model_18_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_18_m_0_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_459 = alloc_buf(56448);
    cpu_pad(buf__model_18_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_459, 40, 40, 32, INPUT_ZP);
    execute_fpga_layer(pad_459, buf__model_18_m_0_cv2_conv_Conv_output_0, wt_phy[459], bias_phy[459], 42, 42, 32, 64, 3, 1, layer_scales[459]);
    cpu_silu(buf__model_18_m_0_cv2_conv_Conv_output_0, 102400, layer_scales[459], layer_zps[459]);

    // --- Node 464: Concat ---
    int8_t* buf__model_18_Split_output_0 = alloc_buf(102400);
    int8_t* buf__model_18_m_0_cv2_act_Mul_output_0 = alloc_buf(102400);
    int8_t* buf__model_18_Concat_output_0 = alloc_buf(307200);
    cpu_concat_append(buf__model_18_Split_output_0, 64, buf__model_18_Concat_output_0, 0, 40, 40);
    cpu_concat_append(buf__model_18_Split_output_1_DequantizeLinear_Output, 64, buf__model_18_Concat_output_0, 64, 40, 40);
    cpu_concat_append(buf__model_18_m_0_cv2_act_Mul_output_0, 64, buf__model_18_Concat_output_0, 128, 40, 40);

    // --- Node 467: Conv ---
    int8_t* buf__model_18_Concat_output_0_DequantizeLinear_Output = alloc_buf(307200);
    int8_t* buf__model_18_cv2_conv_Conv_output_0 = alloc_buf(204800);
    int8_t* pad_467 = alloc_buf(338688);
    cpu_pad(buf__model_18_Concat_output_0_DequantizeLinear_Output, pad_467, 40, 40, 192, INPUT_ZP);
    execute_fpga_layer(pad_467, buf__model_18_cv2_conv_Conv_output_0, wt_phy[467], bias_phy[467], 42, 42, 192, 128, 3, 1, layer_scales[467]);
    cpu_silu(buf__model_18_cv2_conv_Conv_output_0, 204800, layer_scales[467], layer_zps[467]);

    // --- Node 474: Conv ---
    int8_t* buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(204800);
    int8_t* buf__model_19_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_474 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_474, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_474, buf__model_19_conv_Conv_output_0, wt_phy[474], bias_phy[474], 42, 42, 128, 128, 3, 2, layer_scales[474]);
    cpu_silu(buf__model_19_conv_Conv_output_0, 51200, layer_scales[474], layer_zps[474]);

    // --- Node 475: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_475 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_475, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_475, buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0, wt_phy[475], bias_phy[475], 42, 42, 128, 64, 3, 1, layer_scales[475]);
    cpu_silu(buf__model_22_cv2_1_cv2_1_0_conv_Conv_output_0, 102400, layer_scales[475], layer_zps[475]);

    // --- Node 476: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_476 = alloc_buf(225792);
    cpu_pad(buf__model_18_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_476, 40, 40, 128, INPUT_ZP);
    execute_fpga_layer(pad_476, buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0, wt_phy[476], bias_phy[476], 42, 42, 128, 80, 3, 1, layer_scales[476]);
    cpu_silu(buf__model_22_cv3_1_cv3_1_0_conv_Conv_output_0, 128000, layer_scales[476], layer_zps[476]);

    // --- Node 489: Concat ---
    int8_t* buf__model_19_act_Mul_output_0 = alloc_buf(51200);
    int8_t* buf__model_20_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_19_act_Mul_output_0, 128, buf__model_20_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_9_cv2_act_Mul_output_0, 256, buf__model_20_Concat_output_0, 128, 20, 20);

    // --- Node 496: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_496 = alloc_buf(112896);
    cpu_pad(buf__model_22_cv2_1_cv2_1_0_act_Mul_output_0_DequantizeLinear_Output, pad_496, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_496, buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0, wt_phy[496], bias_phy[496], 42, 42, 64, 64, 3, 1, layer_scales[496]);
    cpu_silu(buf__model_22_cv2_1_cv2_1_1_conv_Conv_output_0, 102400, layer_scales[496], layer_zps[496]);

    // --- Node 497: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(128000);
    int8_t* buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_497 = alloc_buf(141120);
    cpu_pad(buf__model_22_cv3_1_cv3_1_0_act_Mul_output_0_DequantizeLinear_Output, pad_497, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(pad_497, buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0, wt_phy[497], bias_phy[497], 42, 42, 80, 80, 3, 1, layer_scales[497]);
    cpu_silu(buf__model_22_cv3_1_cv3_1_1_conv_Conv_output_0, 128000, layer_scales[497], layer_zps[497]);

    // --- Node 498: Conv ---
    int8_t* buf__model_20_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_21_cv1_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_498 = alloc_buf(185856);
    cpu_pad(buf__model_20_Concat_output_0_DequantizeLinear_Output, pad_498, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_498, buf__model_21_cv1_conv_Conv_output_0, wt_phy[498], bias_phy[498], 22, 22, 384, 256, 3, 1, layer_scales[498]);
    cpu_silu(buf__model_21_cv1_conv_Conv_output_0, 102400, layer_scales[498], layer_zps[498]);

    // --- Node 517: Conv ---
    int8_t* buf__model_22_cv2_1_cv2_1_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_1_cv2_1_2_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_517 = alloc_buf(112896);
    cpu_pad(buf__model_22_cv2_1_cv2_1_1_act_Mul_output_0_DequantizeLinear_Output, pad_517, 40, 40, 64, INPUT_ZP);
    execute_fpga_layer(pad_517, buf__model_22_cv2_1_cv2_1_2_Conv_output_0, wt_phy[517], bias_phy[517], 42, 42, 64, 64, 3, 1, layer_scales[517]);
    cpu_silu(buf__model_22_cv2_1_cv2_1_2_Conv_output_0, 102400, layer_scales[517], layer_zps[517]);

    // --- Node 518: Conv ---
    int8_t* buf__model_22_cv3_1_cv3_1_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(128000);
    int8_t* buf__model_22_cv3_1_cv3_1_2_Conv_output_0 = alloc_buf(128000);
    int8_t* pad_518 = alloc_buf(141120);
    cpu_pad(buf__model_22_cv3_1_cv3_1_1_act_Mul_output_0_DequantizeLinear_Output, pad_518, 40, 40, 80, INPUT_ZP);
    execute_fpga_layer(pad_518, buf__model_22_cv3_1_cv3_1_2_Conv_output_0, wt_phy[518], bias_phy[518], 42, 42, 80, 80, 3, 1, layer_scales[518]);
    cpu_silu(buf__model_22_cv3_1_cv3_1_2_Conv_output_0, 128000, layer_scales[518], layer_zps[518]);

    // --- Node 522: Conv ---
    int8_t* buf__model_21_Split_output_1_DequantizeLinear_Output = alloc_buf(51200);
    int8_t* buf__model_21_m_0_cv1_conv_Conv_output_0 = alloc_buf(28800);
    int8_t* pad_522 = alloc_buf(61952);
    cpu_pad(buf__model_21_Split_output_1_DequantizeLinear_Output, pad_522, 20, 20, 128, INPUT_ZP);
    execute_fpga_layer(pad_522, buf__model_21_m_0_cv1_conv_Conv_output_0, wt_phy[522], bias_phy[522], 22, 22, 128, 72, 3, 1, layer_scales[522]);
    cpu_silu(buf__model_21_m_0_cv1_conv_Conv_output_0, 28800, layer_scales[522], layer_zps[522]);

    // --- Node 533: Conv ---
    int8_t* buf__model_21_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(28800);
    int8_t* buf__model_21_m_0_cv2_conv_Conv_output_0 = alloc_buf(51200);
    int8_t* pad_533 = alloc_buf(34848);
    cpu_pad(buf__model_21_m_0_cv1_act_Mul_output_0_DequantizeLinear_Output, pad_533, 20, 20, 72, INPUT_ZP);
    execute_fpga_layer(pad_533, buf__model_21_m_0_cv2_conv_Conv_output_0, wt_phy[533], bias_phy[533], 22, 22, 72, 128, 3, 1, layer_scales[533]);
    cpu_silu(buf__model_21_m_0_cv2_conv_Conv_output_0, 51200, layer_scales[533], layer_zps[533]);

    // --- Node 538: Concat ---
    int8_t* buf__model_21_Split_output_0 = alloc_buf(51200);
    int8_t* buf__model_21_m_0_cv2_act_Mul_output_0 = alloc_buf(51200);
    int8_t* buf__model_21_Concat_output_0 = alloc_buf(153600);
    cpu_concat_append(buf__model_21_Split_output_0, 128, buf__model_21_Concat_output_0, 0, 20, 20);
    cpu_concat_append(buf__model_21_Split_output_1_DequantizeLinear_Output, 128, buf__model_21_Concat_output_0, 128, 20, 20);
    cpu_concat_append(buf__model_21_m_0_cv2_act_Mul_output_0, 128, buf__model_21_Concat_output_0, 256, 20, 20);

    // --- Node 541: Conv ---
    int8_t* buf__model_21_Concat_output_0_DequantizeLinear_Output = alloc_buf(153600);
    int8_t* buf__model_21_cv2_conv_Conv_output_0 = alloc_buf(102400);
    int8_t* pad_541 = alloc_buf(185856);
    cpu_pad(buf__model_21_Concat_output_0_DequantizeLinear_Output, pad_541, 20, 20, 384, INPUT_ZP);
    execute_fpga_layer(pad_541, buf__model_21_cv2_conv_Conv_output_0, wt_phy[541], bias_phy[541], 22, 22, 384, 256, 3, 1, layer_scales[541]);
    cpu_silu(buf__model_21_cv2_conv_Conv_output_0, 102400, layer_scales[541], layer_zps[541]);

    // --- Node 548: Conv ---
    int8_t* buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(102400);
    int8_t* buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_548 = alloc_buf(123904);
    cpu_pad(buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_548, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_548, buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0, wt_phy[548], bias_phy[548], 22, 22, 256, 64, 3, 1, layer_scales[548]);
    cpu_silu(buf__model_22_cv2_2_cv2_2_0_conv_Conv_output_0, 25600, layer_scales[548], layer_zps[548]);

    // --- Node 549: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_549 = alloc_buf(123904);
    cpu_pad(buf__model_21_cv2_act_Mul_output_0_DequantizeLinear_Output, pad_549, 20, 20, 256, INPUT_ZP);
    execute_fpga_layer(pad_549, buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0, wt_phy[549], bias_phy[549], 22, 22, 256, 80, 3, 1, layer_scales[549]);
    cpu_silu(buf__model_22_cv3_2_cv3_2_0_conv_Conv_output_0, 32000, layer_scales[549], layer_zps[549]);

    // --- Node 562: Conv ---
    int8_t* buf__model_22_cv2_2_cv2_2_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(25600);
    int8_t* buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_562 = alloc_buf(30976);
    cpu_pad(buf__model_22_cv2_2_cv2_2_0_act_Mul_output_0_DequantizeLinear_Output, pad_562, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(pad_562, buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0, wt_phy[562], bias_phy[562], 22, 22, 64, 64, 3, 1, layer_scales[562]);
    cpu_silu(buf__model_22_cv2_2_cv2_2_1_conv_Conv_output_0, 25600, layer_scales[562], layer_zps[562]);

    // --- Node 563: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_0_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(32000);
    int8_t* buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_563 = alloc_buf(38720);
    cpu_pad(buf__model_22_cv3_2_cv3_2_0_act_Mul_output_0_DequantizeLinear_Output, pad_563, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(pad_563, buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0, wt_phy[563], bias_phy[563], 22, 22, 80, 80, 3, 1, layer_scales[563]);
    cpu_silu(buf__model_22_cv3_2_cv3_2_1_conv_Conv_output_0, 32000, layer_scales[563], layer_zps[563]);

    // --- Node 576: Conv ---
    int8_t* buf__model_22_cv2_2_cv2_2_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(25600);
    int8_t* buf__model_22_cv2_2_cv2_2_2_Conv_output_0 = alloc_buf(25600);
    int8_t* pad_576 = alloc_buf(30976);
    cpu_pad(buf__model_22_cv2_2_cv2_2_1_act_Mul_output_0_DequantizeLinear_Output, pad_576, 20, 20, 64, INPUT_ZP);
    execute_fpga_layer(pad_576, buf__model_22_cv2_2_cv2_2_2_Conv_output_0, wt_phy[576], bias_phy[576], 22, 22, 64, 64, 3, 1, layer_scales[576]);
    cpu_silu(buf__model_22_cv2_2_cv2_2_2_Conv_output_0, 25600, layer_scales[576], layer_zps[576]);

    // --- Node 577: Conv ---
    int8_t* buf__model_22_cv3_2_cv3_2_1_act_Mul_output_0_DequantizeLinear_Output = alloc_buf(32000);
    int8_t* buf__model_22_cv3_2_cv3_2_2_Conv_output_0 = alloc_buf(32000);
    int8_t* pad_577 = alloc_buf(38720);
    cpu_pad(buf__model_22_cv3_2_cv3_2_1_act_Mul_output_0_DequantizeLinear_Output, pad_577, 20, 20, 80, INPUT_ZP);
    execute_fpga_layer(pad_577, buf__model_22_cv3_2_cv3_2_2_Conv_output_0, wt_phy[577], bias_phy[577], 22, 22, 80, 80, 3, 1, layer_scales[577]);
    cpu_silu(buf__model_22_cv3_2_cv3_2_2_Conv_output_0, 32000, layer_scales[577], layer_zps[577]);

    std::cout << "Total CMA Memory Used: " << current_offset / (1024*1024) << " MB" << std::endl;

    // --- POST PROCESSING PIPELINE ---
    std::cout << "Decoding YOLOv8 DFL Head..." << std::endl;
    std::vector<cv::Rect> final_boxes;
    std::vector<float> final_confs;
    std::vector<int> final_class_ids;

    decode_yolo_branch(buf__model_22_cv2_0_cv2_0_2_Conv_output_0, buf__model_22_cv3_0_cv3_0_2_Conv_output_0,
                       layer_scales[443], layer_scales[444], 8, 80, final_boxes, final_confs, final_class_ids, pad_w, pad_h, scale_factor);

    decode_yolo_branch(buf__model_22_cv2_1_cv2_1_2_Conv_output_0, buf__model_22_cv3_1_cv3_1_2_Conv_output_0,
                       layer_scales[517], layer_scales[518], 16, 40, final_boxes, final_confs, final_class_ids, pad_w, pad_h, scale_factor);

    decode_yolo_branch(buf__model_22_cv2_2_cv2_2_2_Conv_output_0, buf__model_22_cv3_2_cv3_2_2_Conv_output_0,
                       layer_scales[576], layer_scales[577], 32, 20, final_boxes, final_confs, final_class_ids, pad_w, pad_h, scale_factor);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(final_boxes, final_confs, 0.25f, 0.45f, indices);
    for (int idx : indices) {
        cv::rectangle(img, final_boxes[idx], cv::Scalar(0, 255, 0), 2);
    }
}

// =====================================================================
// MAIN ENTRY
// =====================================================================
int main() {
    if (!init_hardware()) return -1;

    std::cout << "Loading weights from memory_map.txt..." << std::endl;
    load_weights("memory_map.txt");

    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load test.jpg" << std::endl;
        return -1;
    }
    
    float scale; int pw, ph;
    std::cout << "Pre-processing image..." << std::endl;
    preprocess(img, cma_buffer.virt_addr, 0.00392f, 0, scale, pw, ph);
    
    std::cout << "Running FPGA Network..." << std::endl;
    run_network(cma_buffer.virt_addr, scale, pw, ph, img);

    cv::imwrite("output.jpg", img);
    std::cout << "Detection complete! Saved to output.jpg" << std::endl;
    
    return 0;
}