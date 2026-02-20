from ultralytics import YOLO
import torch

# --- CONFIG ---
PRUNED_MODEL_PATH = "runs/detect/train/weights/best.pt" 
DATASET = "coco128.yaml"

def test_accuracy():
    print(f"🔎 Testing Accuracy for: {PRUNED_MODEL_PATH}")
    
    try:
        # 1. Load the pruned weights
        pruned_weights = torch.load(PRUNED_MODEL_PATH, weights_only=False)
        
        # 2. Create wrapper
        model = YOLO("yolov8n.pt")
        
        # 3. Inject the pruned brain
        if isinstance(pruned_weights, dict) and 'model' in pruned_weights:
            model.model = pruned_weights['model']
        else:
            model.model = pruned_weights

        # --- THE FIX IS HERE ---
        # Force the entire model to Float32 immediately.
        # This prevents the "Half != Float" error during fusion.
        model.model.float() 
        # -----------------------

        print("✅ Model loaded and cast to Float32.")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. Run Validation
    # We add 'half=False' to be doubly sure it doesn't try to auto-convert back.
    print(f"🚀 Running Validation on {DATASET}...")
    metrics = model.val(data=DATASET, imgsz=640, verbose=False, half=False)
    
    # 5. Extract Metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    params = sum(p.numel() for p in model.model.parameters())

    print("\n" + "="*40)
    print("📊 PRUNED MODEL REPORT")
    print("="*40)
    print(f"Parameters:   {params:,}")
    print(f"mAP (50):     {map50:.4f}")
    print(f"mAP (50-95):  {map50_95:.4f}")
    print("-" * 40)
    
    if map50_95 > 0.35:
        print("✅ STATUS: EXCELLENT (Ready for FPGA)")
    elif map50_95 > 0.25:
        print("⚠️ STATUS: ACCEPTABLE (Might miss small objects)")
    else:
        print("❌ STATUS: NEEDS MORE TRAINING (Too much accuracy lost)")

if __name__ == "__main__":
    test_accuracy()