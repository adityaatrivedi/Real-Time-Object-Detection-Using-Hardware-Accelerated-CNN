from ultralytics import YOLO
import torch

# --- CONFIG ---
# This is where the Trainer saved your successful run
CHECKPOINT_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_ONNX = "yolov8n_final_pruned.onnx"

def export_final():
    print(f"Loading healed checkpoint: {CHECKPOINT_PATH}...")
    
    # 1. Load the checkpoint
    # We use the standard YOLO wrapper because the checkpoint 
    # now contains enough info to load (mostly).
    try:
        model = YOLO(CHECKPOINT_PATH)
    except Exception:
        # Fallback: Load raw if wrapper fails
        raw = torch.load(CHECKPOINT_PATH, weights_only=False)
        model = YOLO("yolov8n.pt")
        model.model = raw['model'] if 'model' in raw else raw

    # 2. Verify Size (One last check)
    params = sum(p.numel() for p in model.model.parameters())
    print(f"Model Params: {params:,}")
    if params > 3000000:
        print("❌ WARNING: This looks like the unpruned model. Check your path.")
        return

    # 3. Patch the Metadata (The Fix)
    # The exporter needs to know the task and dataset to name the outputs
    if not hasattr(model.model, 'args'):
        model.model.args = {}
        
    # We manually inject the keys that caused the crash
    patch_args = {
        'task': 'detect',
        'imgsz': 640,
        'data': 'coco128.yaml',
        'batch': 1,
        'half': False,
        'int8': False,
        'dynamic': False
    }
    
    # Merge patch into existing args
    if isinstance(model.model.args, dict):
        model.model.args.update(patch_args)
    else:
        model.model.args = patch_args

    # 4. Export
    print("🚀 Exporting to ONNX...")
    model.export(format="onnx", opset=11)
    
    # 5. Rename/Move
    import os
    # The export usually saves adjacent to the .pt file
    default_export_path = CHECKPOINT_PATH.replace(".pt", ".onnx")
    
    if os.path.exists(default_export_path):
        os.rename(default_export_path, OUTPUT_ONNX)
        print(f"🎉 SUCCESS! Final FPGA Model: {OUTPUT_ONNX}")
    else:
        print(f"⚠️ Export finished, but file is at: {default_export_path}")

if __name__ == "__main__":
    export_final()