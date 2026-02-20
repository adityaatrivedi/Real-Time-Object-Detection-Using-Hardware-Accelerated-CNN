import onnx
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import os

# Define the input and output model paths
input_model_path = "yolov8n.onnx"
output_model_path = "yolov8n_quantized.onnx"

class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_shape, data_size=10):
        self.input_name = input_name
        self.input_shape = input_shape
        self.data_size = data_size
        self.enum_data_dicts = iter(
            [{self.input_name: np.random.rand(*self.input_shape).astype(np.float32)} 
             for _ in range(self.data_size)]
        )

    def get_next(self):
        return next(self.enum_data_dicts, None)

def quantize_model():
    print(f"Quantizing {input_model_path}...")
    
    # 1. Get input name and shape from the model
    model = onnx.load(input_model_path)
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    # YOLOv8n standard input shape is (1, 3, 640, 640)
    input_shape = (1, 3, 640, 640) 

    # 2. Set up the data reader (Simulating calibration data)
    dr = RandomDataReader(input_name, input_shape)

    # 3. Run Quantization
    quantize_static(
        input_model_path,
        output_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,  # Standard for Vitis AI / Hardware accelerators
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    print(f"Success! Quantized model saved to: {output_model_path}")

if __name__ == "__main__":
    quantize_model()