

YOLOv8 Inference Engine on Xilinx Zynq UltraScale+ MPSoC

Project Overview

This project implements a high-performance inference engine for a custom-trained YOLOv8 object detection model, deployed on a Xilinx ZCU104 Evaluation Board featuring a Zynq UltraScale+ MPSoC. The engine is developed in C++ and leverages the MPSoC’s programmable logic (PL) for hardware-accelerated inference, utilizing AXI DMA for efficient data transfers between the processing system (PS) and PL.

The core of the system involves:
	•	Optimized Model Deployment: A YOLOv8 model is quantized using a minmax strategy, with its binary weights (.bin files for scales and zero points) directly loaded into the DDR memory for access by custom hardware accelerators on the PL.
	•	Hardware Acceleration: A custom AXI-Lite controlled, AXI DMA-interfaced IP core (simulated by app.cpp’s architecture) on the PL performs the heavy computational lifting of the neural network layers.
	•	Efficient Data Handling: OpenCV 4 is used on the PS for robust image preprocessing (letterboxing, resizing, normalization, and quantization) and post-processing (bounding box decoding, confidence thresholding, and Non-Maximum Suppression (NMS)).
	•	Embedded Linux Environment: The application runs on an embedded Linux environment built with the PetaLinux SDK, demonstrating a full-stack edge AI deployment pipeline.

This setup achieves low-latency object detection by offloading critical inference tasks to dedicated hardware, making it suitable for real-time edge AI applications.

⸻

Hardware & Software Requirements

To build, deploy, and run this project, you will need:

Hardware
	•	Xilinx ZCU104 Evaluation Board: The target hardware for deployment (Zynq UltraScale+ MPSoC).
	•	SD Card (8GB or larger): Required for booting the ZCU104 with the PetaLinux image and storing the application.
	•	USB-to-Serial Adapter/Cable: For UART communication with the ZCU104.

Software
	•	Xilinx PetaLinux SDK: Used for building the embedded Linux image, cross-compiling the application, and managing the hardware platform.
	•	OpenCV 4.x Development Libraries: The application relies on OpenCV for image processing tasks (pre- and post-processing). Ensure the version is compatible with your PetaLinux environment and toolchain.
	•	Host Machine: A Linux-based operating system (e.g., Ubuntu) with the PetaLinux SDK installed.
	•	dd utility: For flashing the SD card image.

⸻

Repository Structure

The project directory is organized as follows:
	•	app.cpp: Contains the low-level AXI DMA and AXI Lite control logic for interacting with the custom YOLO acceleration IP on the Zynq’s Programmable Logic (PL). This file orchestrates data transfers and IP configuration for sequential layer processing.
	•	main.cpp: The main application logic. This file handles:
	•	Image preprocessing (letterboxing, resizing, BGR-to-RGB conversion, and UINT8 quantization for FPGA input) using OpenCV.
	•	Calling the hardware scheduler (implicitly using the logic from app.cpp to offload inference to the PL).
	•	Post-processing of FPGA output, including bounding box decoding, confidence thresholding, NMS (Non-Maximum Suppression) with OpenCV’s cv::dnn::NMSBoxes, and drawing results on the output image.
	•	binary_weights_final/: This directory stores the minmax quantized binary weights and corresponding scale/zero-point values (.bin files) for each layer of the YOLOv8 model. These files are loaded into DDR memory and accessed by the custom hardware accelerator.
	•	calibration_images/: Contains images used for post-training quantization calibration.
	•	yolov8-mobilenetv3.yaml: Likely the model definition file for the custom YOLOv8 architecture using MobileNetV3 backbone.
	•	*.onnx files: Various ONNX model exports, likely representing different stages of quantization or pruning (yolov8n_entropy.onnx, yolov8n_final_int8_minmax.onnx, etc.).
	•	*.pt files: PyTorch model checkpoints (yolov3-tiny.pt, yolov8n.pt, etc.).
	•	*.py files: A collection of Python scripts for tasks such as:
	•	export_onnx.py, export_final.py: Exporting models to ONNX format.
	•	quantize.py, quantize_real.py, tune_quantization.py, final_safe_quantize.py, hybrid_quantize.py: Scripts related to model quantization.
	•	prune_model.py, prune_final.py, finetune_pruned.py: Scripts for model pruning and fine-tuning.
	•	extract_weights_onnx.py, extract_all_params.py: Utilities for extracting model parameters/weights.
	•	test_onnx.py, test_final_accuracy.py, compare_accuracy.py, verify_quantized.py: Scripts for model validation and accuracy comparison.
	•	train_mobilenet.py, train_fixed.py: Training scripts.

⸻

Cross-Compilation Steps

The application needs to be cross-compiled for the ARM architecture of the Zynq UltraScale+ MPSoC. This process assumes you have sourced your PetaLinux SDK environment (e.g., source <path-to-petalinux>/settings.sh).
	1.	Prepare the environment:

source /opt/pkg/petalinux/2023.2/settings.sh
aarch64-linux-gnu-g++ --version


	2.	Compile the C++ application:

SYSROOT_PATH=<PATH_TO_PETALINUX_SYSROOT>/sysroots/aarch64-xilinx-linux

aarch64-linux-gnu-g++ \
    main.cpp app.cpp \
    -o yolo_app \
    -std=c++11 \
    -I${SYSROOT_PATH}/usr/include/opencv4 \
    -I${SYSROOT_PATH}/usr/include/xilinx-uio \
    -I<PATH_TO_XILINX_VIVADO_HLS_IP_DRIVER_HEADERS> \
    --sysroot=${SYSROOT_PATH} \
    -L${SYSROOT_PATH}/usr/lib \
    -L${SYSROOT_PATH}/lib \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn \
    -lxil \
    -lpthread \
    -lm \
    -Wl,-rpath-link=${SYSROOT_PATH}/usr/lib \
    -Wl,-rpath-link=${SYSROOT_PATH}/lib

	•	Adjust include and library paths based on your PetaLinux build.
	•	--sysroot ensures headers and libraries from the target system are used instead of the host system.

Upon successful compilation, an executable named yolo_app will be generated.
