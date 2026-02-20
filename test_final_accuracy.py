from ultralytics import YOLO
import pandas as pd

# --- CONFIG ---
MODEL_PATH = "yolov8n_final_int8_minmax.onnx"
DATASET = "coco128.yaml"

def test_final():
    print(f"🚀 Loading Quantized Model: {MODEL_PATH}...")
    
    # Load the ONNX model into the YOLO wrapper
    # We specify 'task=detect' to ensure it knows how to interpret the outputs
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        print(f"❌ Error loading ONNX: {e}")
        return

    # Run Validation
    print("running validation on COCO128 (this uses ONNXRuntime)...")
    # imgsz=640 is critical because the ONNX model has a fixed input size
    metrics = model.val(data=DATASET, imgsz=640, verbose=False)
    
    # Extract Scores
    map50 = metrics.box.map50
    map50_95 = metrics.box.map  # This is the standard "mAP"
    
    # --- REPORT CARD ---
    print("\n" + "="*50)
    print("📊 FINAL FPGA MODEL REPORT CARD")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"mAP (50-95): {map50_95:.4f}")
    print(f"mAP (50):    {map50:.4f}")
    print("-" * 50)
    
    # Comparison Logic
    print("COMPARISON:")
    print("1. Original YOLOv8n:    ~0.4537  (Baseline)")
    print("2. Pruned (Float32):    ~0.3870  (After Healing)")
    print(f"3. Quantized (INT8):    {map50_95:.4f}  (Current)")
    
    drop = 0.3870 - map50_95
    if drop < 0.02:
        print(f"\n✅ SUCCESS: Minimal accuracy loss ({drop:.4f}). Excellent for FPGA!")
    elif drop < 0.05:
        print(f"\n⚠️ ACCEPTABLE: Moderate loss ({drop:.4f}). Good trade-off for speed.")
    else:
        print(f"\n❌ WARNING: High accuracy loss ({drop:.4f}). Consider Post-Training Quantization (PTQ) fine-tuning.")

if __name__ == "__main__":
    test_final()