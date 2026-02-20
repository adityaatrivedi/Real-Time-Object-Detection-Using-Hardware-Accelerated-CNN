#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

// --- INPUT QUANTIZATION CONFIG ---
// Copied directly from the first QuantizeLinear node in Netron
const float INPUT_SCALE = 0.003921568859368563f;      
const int8_t INPUT_ZERO_POINT = 0;       

// --- POST-PROCESSING CONFIG ---
const float CONF_THRESHOLD = 0.25f;      // Ignore boxes below 25% confidence
const float NMS_THRESHOLD = 0.45f;       // IOU threshold for overlapping boxes

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// ---------------------------------------------------------
// STEP 2: PRE-PROCESSING (LETTERBOX & QUANTIZE)
// ---------------------------------------------------------
void preprocess_fpga_input(const cv::Mat& original_image, uint8_t* fpga_input_memory, 
                           float& scale_factor, int& pad_w, int& pad_h) {
    
    int target_size = 640;
    int img_w = original_image.cols;
    int img_h = original_image.rows;

    // 1. Calculate the Scaling Factor (Keep Aspect Ratio)
    scale_factor = std::min((float)target_size / img_w, (float)target_size / img_h);
    
    int new_unpad_w = std::round(img_w * scale_factor);
    int new_unpad_h = std::round(img_h * scale_factor);

    // 2. Resize the image
    cv::Mat resized_img;
    cv::resize(original_image, resized_img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    // 3. Calculate Padding (To reach exactly 640x640)
    float dw = (target_size - new_unpad_w) / 2.0f;
    float dh = (target_size - new_unpad_h) / 2.0f;

    int top = std::round(dh - 0.1f);
    int bottom = std::round(dh + 0.1f);
    int left = std::round(dw - 0.1f);
    int right = std::round(dw + 0.1f);

    // Save the integer padding so Step 5 can use it to reverse the math
    pad_w = left;
    pad_h = top;

    // 4. Apply the Border 
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 5. Convert BGR (OpenCV default) to RGB (YOLO default)
    cv::cvtColor(padded_img, padded_img, cv::COLOR_BGR2RGB);

    // 6. Quantize and flatten into FPGA Memory (CHW format)
    int channels = 3;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < target_size; h++) {
            for (int w = 0; w < target_size; w++) {
                
                uint8_t pixel_val = padded_img.at<cv::Vec3b>(h, w)[c];
                float float_val = pixel_val / 255.0f;

                // Apply Quantization Formula
                int quantized_val = std::round(float_val / INPUT_SCALE) + INPUT_ZERO_POINT;

                // FIXED: Clamp to UINT8 range (0 to 255)
                if (quantized_val > 255) quantized_val = 255;
                if (quantized_val < 0) quantized_val = 0;

                int memory_index = (c * target_size * target_size) + (h * target_size) + w;
                fpga_input_memory[memory_index] = (uint8_t)quantized_val;
            }
        }
    }
    
    std::cout << "Image pre-processed and loaded into input buffer." << std::endl;
}

// ---------------------------------------------------------
// STEP 4 & 5: READ FLOAT ARRAY, NMS, AND SCALE
// ---------------------------------------------------------
// FIXED: Input is now a float array, skipping manual dequantization
void process_fpga_output(float* fpga_memory, cv::Mat& original_image, 
                         float scale_factor, int pad_w, int pad_h) {
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // We loop through all 8,400 columns (predictions)
    for (int col = 0; col < 8400; col++) {
        
        float max_conf = 0.0f;
        int best_class_id = -1;

        // 1. Find the highest class score for this column (Rows 4 to 83)
        for (int row = 4; row < 84; row++) {
            int index = (row * 8400) + col; 
            
            // Values are already floats!
            float conf = fpga_memory[index];
            
            if (conf > max_conf) {
                max_conf = conf;
                best_class_id = row - 4; // Shift index back to 0-79
            }
        }

        // 2. Thresholding: If it's a confident prediction, extract the box
        if (max_conf >= CONF_THRESHOLD) {
            
            // Extract box coordinates (Rows 0 to 3)
            float cx = fpga_memory[(0 * 8400) + col];
            float cy = fpga_memory[(1 * 8400) + col];
            float w  = fpga_memory[(2 * 8400) + col];
            float h  = fpga_memory[(3 * 8400) + col];

            // Convert Center (cx, cy) to Top-Left (x, y) for OpenCV
            float x_top_left = cx - (w / 2.0f);
            float y_top_left = cy - (h / 2.0f);

            // Reverse the Pre-Processing padding and scaling
            int final_x = std::round((x_top_left - pad_w) / scale_factor);
            int final_y = std::round((y_top_left - pad_h) / scale_factor);
            int final_w = std::round(w / scale_factor);
            int final_h = std::round(h / scale_factor);

            boxes.push_back(cv::Rect(final_x, final_y, final_w, final_h));
            confidences.push_back(max_conf);
            class_ids.push_back(best_class_id);
        }
    }

    // STEP 5: Post-Processing (Non-Maximum Suppression)
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_indices);

    // Draw the final surviving boxes on the original image
    for (int idx : nms_indices) {
        cv::Rect box = boxes[idx];
        int class_id = class_ids[idx];
        float conf = confidences[idx];

        cv::rectangle(original_image, box, cv::Scalar(0, 255, 0), 2);
        
        std::string label = "Class " + std::to_string(class_id) + ": " + std::to_string(conf).substr(0, 4);
        cv::putText(original_image, label, cv::Point(box.x, box.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
    
    std::cout << "Detected " << nms_indices.size() << " objects!" << std::endl;
}

// ---------------------------------------------------------
// MAIN SCHEDULER
// ---------------------------------------------------------
int main() {
    // 1. Setup Memory Pointers 
    uint8_t* fpga_input_buffer = new uint8_t[3 * 640 * 640]; 
    
    // FIXED: The output buffer is allocated as floats
    float* fpga_output_buffer = new float[84 * 8400];    
    
    // Read the camera or file
    cv::Mat raw_camera_feed = cv::imread("test_image.jpg");
    if (raw_camera_feed.empty()) {
        std::cerr << "Error: Could not load test_image.jpg" << std::endl;
        return -1;
    }

    float scale_factor = 0.0f;
    int pad_w = 0, pad_h = 0;

    // --- EXECUTE STEP 2: PRE-PROCESS ---
    preprocess_fpga_input(raw_camera_feed, fpga_input_buffer, 
                          scale_factor, pad_w, pad_h);

    // --- EXECUTE STEP 3: HARDWARE SCHEDULER ---
    std::cout << "Sending START signal to FPGA..." << std::endl;
    
    // ... [Your team's code to map buffers to DDR, toggle AXI registers, and wait for DONE] ...

    std::cout << "FPGA processing complete." << std::endl;

    // --- EXECUTE STEP 4 & 5: POST-PROCESS ---
    process_fpga_output(fpga_output_buffer, raw_camera_feed, 
                        scale_factor, pad_w, pad_h);

    // Show the final result!
    cv::imwrite("final_detection.jpg", raw_camera_feed);
    std::cout << "Done! Saved to final_detection.jpg" << std::endl;

    delete[] fpga_input_buffer;
    delete[] fpga_output_buffer;
    return 0;
}