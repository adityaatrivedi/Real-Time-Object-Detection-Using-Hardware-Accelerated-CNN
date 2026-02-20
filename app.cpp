#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "math.h"  // For sigmoid/exp

// --- CONFIGURATION ---
// These addresses must match your Vivado Address Editor
#define YOLO_IP_BASE_ADDR   XPAR_YOLO_ACCEL_0_S00_AXI_BASEADDR
#define DMA_DEV_ID          XPAR_AXIDMA_0_DEVICE_ID

// --- DDR MEMORY MAP (Step 2) ---
// We reserve specific RAM areas for different things
#define MEM_BASE_ADDR       0x10000000 
#define TX_BUFFER_BASE      (MEM_BASE_ADDR + 0x00100000) // Input Image
#define RX_BUFFER_BASE      (MEM_BASE_ADDR + 0x00300000) // Output Features
#define WEIGHTS_BASE        (MEM_BASE_ADDR + 0x01000000) // Weights

// Register Offsets (Defined in your Verilog AXI Lite wrapper)
#define REG_CONTROL         0x00
#define REG_LAYER_CONFIG    0x04
#define REG_STATUS          0x08

XAxiDma AxiDma;

// --- STEP 10: SOFTWARE POST-PROCESSING ---
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void post_process_results(float* data, int size) {
    xil_printf("PHASE 10: CPU Post-Processing...\n");
    // 1. Decode Boxes
    // 2. Sigmoid
    // 3. NMS
    // Simple example: Print top confidence
    for(int i=0; i<10; i++) {
        float val = sigmoid(data[i]); // Convert logic to probability
        if(val > 0.5) {
            xil_printf("Detected Object at index %d: Conf %d%%\n", i, (int)(val*100));
        }
    }
}

int main() {
    init_platform();
    int Status;

    xil_printf("--- STEP 1: INITIALIZATION ---\n");

    // Initialize DMA
    XAxiDma_Config *CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
    Status = XAxiDma_CfgInitialize(&AxiDma, CfgPtr);
    if (Status != XST_SUCCESS) return XST_FAILURE;

    // Disable Interrupts for Polling Mode (Simpler for now)
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    xil_printf("--- STEP 2: LOAD WEIGHTS ---\n");
    // In a real app, use f_read() from SD card here.
    // For now, we assume weights are pre-loaded or we write dummy data.
    float *weights_ptr = (float *)WEIGHTS_BASE;
    // weights_ptr[0] = ... load from SD ...

    xil_printf("--- STEP 3: GET INPUT IMAGE ---\n");
    // Pointer to input buffer
    u8 *input_image = (u8 *)TX_BUFFER_BASE;
    // Preprocess: Resize/Normalize usually happens here
    // input_image[0] = ... camera data ...

    // IMPORTANT: Flush Cache so DMA sees the data in DDR
    Xil_DCacheFlushRange((UINTPTR)TX_BUFFER_BASE, 640*640*3);
    Xil_DCacheFlushRange((UINTPTR)WEIGHTS_BASE, 1024*1024); // Size depends on model

    // --- STEP 9: LAYER CHAINING LOOP ---
    // Example: Running 3 Layers
    u32 current_input_addr = TX_BUFFER_BASE;
    u32 current_output_addr = RX_BUFFER_BASE;
    
    for (int layer = 0; layer < 3; layer++) {
        xil_printf("Executing Layer %d...\n", layer);

        // --- STEP 4: CONFIGURE YOLO IP ---
        // Tell FPGA which layer configs to use
        Xil_Out32(YOLO_IP_BASE_ADDR + REG_LAYER_CONFIG, layer); 
        
        // Start the IP (Write 1 to Control Register ap_start)
        // Note: Some IPs need to be started AFTER DMA, some BEFORE. Check your IP docs.
        Xil_Out32(YOLO_IP_BASE_ADDR + REG_CONTROL, 1);

        // --- STEP 5: CONFIGURE AXI DMA ---
        
        // 1. S2MM (Receive Channel - Output from FPGA)
        // We start receive FIRST so it's ready to catch data
        u32 output_size = 32 * 32 * 16; // Calculate exact size for this layer!
        Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)current_output_addr,
                                        output_size, XAXIDMA_DEVICE_TO_DMA);

        // 2. MM2S (Send Channel - Input to FPGA)
        u32 input_size = 64 * 64 * 3; // Calculate exact size for this layer!
        Status = XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)current_input_addr,
                                        input_size, XAXIDMA_DMA_TO_DEVICE);

        // --- STEP 6 & 7: WAIT FOR COMPLETION ---
        // Poll the DMA Busy bit
        while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) {
                // Wait for Input transfer to finish
        }
        while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) {
                // Wait for Output transfer to finish
        }
        
        // Also check IP status register if it has a "Done" bit
        // while ( (Xil_In32(YOLO_IP_BASE_ADDR + REG_STATUS) & 0x1) == 0 );

        xil_printf("Layer %d Done.\n", layer);

        // --- STEP 8: POINTER SWAPPING ---
        // The output of this layer becomes the input of the next
        // We ping-pong between two buffers to save memory
        u32 temp = current_input_addr;
        current_input_addr = current_output_addr; 
        current_output_addr = temp; // Reuse the old input buffer for next output
        
        // Invalidate Cache for the new output data so CPU can read it
        Xil_DCacheInvalidateRange((UINTPTR)current_input_addr, output_size);
    }

    // --- STEP 10: FINAL DETECTION ---
    // Read the final feature map from the last output address
    float *final_features = (float *)current_input_addr; // It was swapped to input!
    post_process_results(final_features, 100);

    cleanup_platform();
    return 0;
}