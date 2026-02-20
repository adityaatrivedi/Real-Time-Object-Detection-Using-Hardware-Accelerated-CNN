import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import os
import cv2
import glob

# Settings
input_model = "yolov8n.onnx"
output_model = "yolov8n_quantized.onnx"
calibration_folder = "calibration_images"

class RealDataReader(CalibrationDataReader):
    def __init__(self, input_name, image_folder):
        self.input_name = input_name
        self.image_paths = glob.glob(os.path.join(image_folder, "*"))
        self.enum_data = iter(self.image_paths)
        print(f"Found {len(self.image_paths)} images for calibration.")

    def get_next(self):
        # Loop until we find a valid image or run out of files
        while True:
            try:
                image_path = next(self.enum_data)
                
                # 1. Load Image
                img = cv2.imread(image_path)
                
                # SAFETY CHECK: If image is bad, skip it instead of crashing
                if img is None:
                    print(f"⚠️ Warning: Could not read {image_path}. Skipping...")
                    continue 

                # 2. Resize to 640x640 (Model Input Size)
                img = cv2.resize(img, (640, 640))
                
                # 3. Convert BGR (OpenCV) to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 4. Normalize (0-255 -> 0.0-1.0)
                img = img.astype(np.float32) / 255.0
                
                # 5. Transpose (HWC -> CHW)
                img = img.transpose(2, 0, 1)
                
                # 6. Add Batch Dimension
                img = np.expand_dims(img, axis=0)
                
                # Return the valid data
                return {self.input_name: img}
                
            except StopIteration:
                # No more files left
                return None

def quantize():
    print(f"Quantizing {input_model} with REAL data...")
    
    # Load model to get input name
    model = onnx.load(input_model)
    input_name = model.graph.input[0].name
    
    # Setup Reader
    dr = RealDataReader(input_name, calibration_folder)
    
    # Run Quantization
    quantize_static(
        input_model,
        output_model,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    print(f"Success! Model saved to {output_model}")

if __name__ == "__main__":
    quantize()