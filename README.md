# Bare-Metal YOLOv8 Inference Engine on Xilinx Zynq UltraScale+ MPSoC

## Project Overview

This project implements a high-performance, bare-metal/Linux inference engine for a custom-trained YOLOv8 object detection model, deployed on a Xilinx ZCU104 Evaluation Board featuring a Zynq UltraScale+ MPSoC. The engine is developed in C++ and leverages the MPSoC's programmable logic (PL) for hardware-accelerated inference, utilizing AXI DMA for efficient data transfers between the processing system (PS) and PL.

The core of the system involves:
-   **Optimized Model Deployment:** A YOLOv8 model is quantized using a minmax strategy, with its binary weights (`.bin` files for scales and zero points) directly loaded into the DDR memory for access by custom hardware accelerators on the PL.
-   **Hardware Acceleration:** A custom AXI-Lite controlled, AXI DMA-interfaced IP core (simulated by `app.cpp`'s architecture) on the PL performs the heavy computational lifting of the neural network layers.
-   **Efficient Data Handling:** OpenCV 4 is used on the PS for robust image preprocessing (letterboxing, resizing, normalization, and quantization) and post-processing (bounding box decoding, confidence thresholding, and Non-Maximum Suppression (NMS)).
-   **Bare-Metal Environment:** The application runs on an embedded Linux environment built with the PetaLinux SDK, demonstrating a full-stack edge AI deployment pipeline.

This setup achieves low-latency object detection by offloading critical inference tasks to dedicated hardware, making it suitable for real-time edge AI applications.

## Hardware & Software Requirements

To build, deploy, and run this project, you will need:

### Hardware
*   **Xilinx ZCU104 Evaluation Board:** The target hardware for deployment (Zynq UltraScale+ MPSoC).
*   **SD Card (8GB or larger):** Required for booting the ZCU104 with the PetaLinux image and storing the application.
*   **USB-to-Serial Adapter/Cable:** For UART communication with the ZCU104.

### Software
*   **Xilinx PetaLinux SDK:** Used for building the embedded Linux image, cross-compiling the application, and managing the hardware platform.
*   **OpenCV 4.x Development Libraries:** The application relies on OpenCV for image processing tasks (pre- and post-processing). Ensure the version is compatible with your PetaLinux environment and toolchain.
*   **Host Machine:** A Linux-based operating system (e.g., Ubuntu) with the PetaLinux SDK installed.
*   **`dd` utility:** For flashing the SD card image.

## Repository Structure

The project directory is organized as follows:

*   **`app.cpp`**: Contains the low-level AXI DMA and AXI Lite control logic for interacting with the custom YOLO acceleration IP on the Zynq's Programmable Logic (PL). This file orchestrates data transfers and IP configuration for sequential layer processing.
*   **`main.cpp`**: The main application logic. This file handles:
    *   Image preprocessing (letterboxing, resizing, BGR-to-RGB conversion, and UINT8 quantization for FPGA input) using OpenCV.
    *   Calling the hardware scheduler (implicitly using the logic from `app.cpp` to offload inference to the PL).
    *   Post-processing of FPGA output, including bounding box decoding, confidence thresholding, NMS (Non-Maximum Suppression) with OpenCV's `cv::dnn::NMSBoxes`, and drawing results on the output image.
*   **`binary_weights_final/`**: This directory stores the minmax quantized binary weights and corresponding scale/zero-point values (`.bin` files) for each layer of the YOLOv8 model. These files are loaded into DDR memory and accessed by the custom hardware accelerator.
*   **`calibration_images/`**: Contains images used for post-training quantization calibration.
*   **`yolov8-mobilenetv3.yaml`**: Likely the model definition file for the custom YOLOv8 architecture using MobileNetV3 backbone.
*   **`*.onnx` files**: Various ONNX model exports, likely representing different stages of quantization or pruning (`yolov8n_entropy.onnx`, `yolov8n_final_int8_minmax.onnx`, etc.).
*   **`*.pt` files**: PyTorch model checkpoints (`yolov3-tiny.pt`, `yolov8n.pt`, etc.).
*   **`*.py` files**: A collection of Python scripts for tasks such as:
    *   `export_onnx.py`, `export_final.py`: Exporting models to ONNX format.
    *   `quantize.py`, `quantize_real.py`, `tune_quantization.py`, `final_safe_quantize.py`, `hybrid_quantize.py`: Scripts related to model quantization.
    *   `prune_model.py`, `prune_final.py`, `finetune_pruned.py`: Scripts for model pruning and fine-tuning.
    *   `extract_weights_onnx.py`, `extract_all_params.py`: Utilities for extracting model parameters/weights.
    *   `test_onnx.py`, `test_final_accuracy.py`, `compare_accuracy.py`, `verify_quantized.py`: Scripts for model validation and accuracy comparison.
    *   `train_mobilenet.py`, `train_fixed.py`: Training scripts.

## Cross-Compilation Steps

The application needs to be cross-compiled for the ARM architecture of the Zynq UltraScale+ MPSoC. This process assumes you have sourced your PetaLinux SDK environment (e.g., `source <path-to-petalinux>/settings.sh`).

1.  **Prepare the environment:**
    Ensure your PetaLinux environment is sourced. This typically sets up the `ARCH`, `CROSS_COMPILE`, `CXX`, and `CC` environment variables, and adds the cross-compiler toolchain to your `PATH`.

    ```bash
    source /opt/pkg/petalinux/2023.2/settings.sh
    # Verify the cross-compiler is available
    aarch64-linux-gnu-g++ --version
    ```

2.  **Compile the C++ application:**
    Navigate to the project root directory and use the `aarch64-linux-gnu-g++` cross-compiler. You will need to link against OpenCV libraries, `xilinxplatform`, `xaxidma`, and other necessary libraries for the embedded system.

    ```bash
    # Example compilation command (adjust include and library paths as per your PetaLinux sysroot)
    # This command compiles main.cpp and app.cpp, links against OpenCV (core, imgcodecs, dnn),
    # Xilinx baremetal libraries (platform, axidma, xil_io, xil_cache), and math.
    # Replace `<PATH_TO_PETALINUX_SYSROOT>` with the actual path to your PetaLinux SDK sysroot.
    # The --sysroot option is crucial for ensuring the compiler uses the target's libraries and headers.

    SYSROOT_PATH=<PATH_TO_PETALINUX_SYSROOT>/sysroots/aarch64-xilinx-linux

    aarch64-linux-gnu-g++ 
        main.cpp app.cpp 
        -o yolo_app 
        -std=c++11 
        -I${SYSROOT_PATH}/usr/include/opencv4 
        -I${SYSROOT_PATH}/usr/include/xilinx-uio 
        -I<PATH_TO_XILINX_VIVADO_HLS_IP_DRIVER_HEADERS> 
        --sysroot=${SYSROOT_PATH} 
        -L${SYSROOT_PATH}/usr/lib 
        -L${SYSROOT_PATH}/lib 
        -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn 
        -lxil 
        -lxil_standalone 
        -lpthread 
        -lm 
        -Wl,-rpath-link=${SYSROOT_PATH}/usr/lib 
        -Wl,-rpath-link=${SYSROOT_PATH}/lib
    ```

    *   **Note:** The exact paths for Xilinx specific headers (like `platform.h`, `xaxidma.h`) and `XILINX_VIVADO_HLS_IP_DRIVER_HEADERS` will depend on your PetaLinux build and Vivado IP integration. You might need to locate these within your PetaLinux build directory or Vivado installation.
    *   `--sysroot` ensures that headers and libraries from the target system (Zynq MPSoC's Linux) are used instead of the host system's.
    *   `-lxil` and `-lxil_standalone` are placeholders for typical Xilinx baremetal/driver libraries. Adjust based on your PetaLinux configuration.
    *   `-lopencv_*` link to the OpenCV libraries.

Upon successful compilation, an executable named `yolo_app` will be generated.

## SD Card Flashing & Deployment

The ZCU104 requires a properly formatted SD card with a bootable Linux image and the application files.

### 1. Prepare the SD Card Image

Your PetaLinux project will generate a WIC (Writeable Image Container) file, typically named something like `petalinux-sdimage.wic`. This image contains both the BOOT partition (FAT32) and the root file system (EXT4).

*   **Identify your SD card device:**
    Insert your SD card into your host machine. Use `lsblk` or `fdisk -l` to identify its device name (e.g., `/dev/sdX` or `/dev/mmcblk0`). **Be extremely careful to select the correct device, as flashing the wrong device can lead to data loss.**

    ```bash
    sudo lsblk
    ```

*   **Flash the WIC image:**
    Unmount any partitions of the SD card before flashing. Replace `/dev/sdX` with your identified SD card device.

    ```bash
    sudo umount /dev/sdX* # Unmount all partitions of the SD card
    sudo dd if=./images/linux/petalinux-sdimage.wic of=/dev/sdX bs=4M status=progress
    sync
    ```
    The `sync` command is crucial to ensure all data is written to the SD card before removal.

### 2. Copy Application Files to the SD Card

After flashing, re-insert the SD card. Your host machine should now recognize two partitions: `BOOT` (FAT32) and `rootfs` (EXT4). If `rootfs` doesn't auto-mount, mount it manually.

*   **Create target directory:**
    It's recommended to create a dedicated directory on the board for your application.

    ```bash
    # On your host machine, assuming /media/user/rootfs is your mounted rootfs partition
    sudo mkdir -p /media/user/rootfs/home/root/yolo_project
    ```

*   **Copy the compiled application and assets:**
    Copy the `yolo_app` executable, your test image (`test_image.jpg`), and the `binary_weights_final` directory to the `yolo_project` directory on the SD card's rootfs.

    ```bash
    sudo cp yolo_app /media/user/rootfs/home/root/yolo_project/
    sudo cp test_image.jpg /media/user/rootfs/home/root/yolo_project/
    sudo cp -r binary_weights_final/ /media/user/rootfs/home/root/yolo_project/
    # If you have an FPGA bitstream to load via the application, copy it as well
    # sudo cp system.bit /media/user/rootfs/home/root/yolo_project/
    sync
    ```

*   **Unmount and Eject:**
    Always unmount the SD card partitions before physically removing it.

    ```bash
    sudo umount /media/user/BOOT /media/user/rootfs
    # Or use: sudo umount /dev/sdX*
    ```
    You can now safely eject the SD card.

## Execution on the Board

Follow these steps to run the YOLOv8 inference application on the ZCU104.

### 1. Hardware Setup

*   **Insert SD Card:** Carefully insert the prepared SD card into the ZCU104's SD card slot.
*   **Set Boot Mode:** Configure the SW6 boot mode switches on the ZCU104 for SD card boot.
    *   **SW6 DIP Switch Settings:**
        *   1: ON
        *   2: OFF
        *   3: OFF
        *   4: OFF
        (All other switches typically OFF for SD boot from JTAG/UART port)
*   **Connect UART:** Connect a USB-to-serial cable between the ZCU104's UART port (often marked J83/J84 for the FTDI chip) and your host PC.
*   **Power On:** Connect the power supply to the ZCU104 and power it on.

### 2. Connect via UART

Open a serial terminal program (e.g., `minicom`, `screen`, `PuTTY`) on your host PC and configure it for the following settings:

*   **Baud Rate:** 115200
*   **Data Bits:** 8
*   **Stop Bits:** 1
*   **Parity:** None
*   **Flow Control:** None

You should see the Linux boot messages in the terminal.

### 3. Log In and Run Application

Once the Linux prompt appears:

*   **Log in:** The default username is `root` with no password.

    ```bash
    root@zcu104:~#
    ```

*   **Navigate to project directory:**

    ```bash
    cd /home/root/yolo_project/
    ```

*   **Run the application:**

    ```bash
    ./yolo_app
    ```

The application will then:
1.  Load `test_image.jpg`.
2.  Pre-process the image for the FPGA.
3.  Initiate the hardware acceleration on the PL.
4.  Receive the processed output from the PL.
5.  Perform CPU-based post-processing (NMS, bounding box scaling).
6.  Save the `final_detection.jpg` with detected objects to the current directory.

You can then retrieve `final_detection.jpg` using `scp` or by copying it back to the SD card's rootfs partition on your host machine to verify the results.

## Future Improvements / To-Do

*   **Integrate IP Core Control:** Refine the `app.cpp` to fully integrate with a custom Vivado HLS generated IP core, ensuring proper AXI-Lite register writes for configuration and AXI DMA transfers.
*   **Dynamic Weight Loading:** Implement robust weight loading from SD card for the `binary_weights_final/` directory, rather than assuming pre-loaded or dummy data.
*   **Performance Optimization:** Profile the application to identify bottlenecks in both PS (OpenCV pre/post-processing) and PL (IP core execution) and optimize for higher FPS. This may include optimizing memory access patterns, parallelizing operations, or improving the IP core's efficiency.
*   **Real-time Video Input:** Integrate with a camera sensor (e.g., MIPI CSI-2) for live video inference.
*   **Error Handling:** Add more comprehensive error handling and reporting for robustness.
*   **Configuration Flexibility:** Externalize configuration parameters (e.g., confidence thresholds, NMS thresholds, model input dimensions) rather than hardcoding them.
*   **Quantization Pipeline Automation:** Streamline the quantization and export process for new YOLOv8 models.