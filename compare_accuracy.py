from ultralytics import YOLO
import pandas as pd

# Define models
models_to_test = {
    "Original (FP32)": "yolov8n.onnx",
    "Quantized (INT8)": "yolov8n_safe_hybrid.onnx",
    "Quantized (Entropy)": "yolov8n_entropy.onnx"
}

# Define Dataset
# 'coco128.yaml' is the standard small dataset (128 images). 
# If you want the TRULY vast version (5000 images), change this to 'coco.yaml'
# (Warning: 'coco.yaml' will download ~1GB of data)
DATASET = 'coco128.yaml'

results_log = []

print(f"🔎 Starting Comparison using {DATASET}...")

for name, path in models_to_test.items():
    print(f"\n--- Testing {name} ---")
    try:
        # Load model
        model = YOLO(path, task='detect')
        
        # Run Validation
        # imgsz=640: Standard size
        # conf=0.25: Standard confidence threshold
        metrics = model.val(data=DATASET, imgsz=640, verbose=False)
        
        # Extract mAP (Mean Average Precision)
        # map50-95 is the most rigorous metric (averages precision over various overlaps)
        map50_95 = metrics.box.map
        map50 = metrics.box.map50
        
        results_log.append({
            "Model": name,
            "mAP (50-95)": round(map50_95, 4),
            "mAP (50)": round(map50, 4)
        })
        print(f"✅ Done. mAP(50-95): {map50_95:.4f}")
        
    except Exception as e:
        print(f"❌ Failed to test {name}: {e}")
        results_log.append({
            "Model": name,
            "mAP (50-95)": 0.0,
            "mAP (50)": 0.0
        })

# --- Final Comparison Report ---
print("\n" + "="*40)
print(f"📊 FINAL ACCURACY REPORT ({DATASET})")
print("="*40)

df = pd.DataFrame(results_log)
print(df.to_string(index=False))

if len(results_log) == 2:
    original = results_log[0]["mAP (50-95)"]
    quantized = results_log[1]["mAP (50-95)"]
    
    if original > 0:
        drop = ((original - quantized) / original) * 100
        print("\n📉 Performance Drop: {:.2f}%".format(drop))
        
        if drop < 3.0:
            print("🚀 STATUS: EXCELLENT (Less than 3% loss)")
        elif drop < 5.0:
            print("⚠️ STATUS: ACCEPTABLE (3-5% loss)")
        else:
            print("❌ STATUS: NEEDS RETUNING (More than 5% loss)")
    else:
        print("\n❌ Error: Original model had 0 accuracy.")