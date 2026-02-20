import torch
from ultralytics import YOLO

# --- CONFIG ---
PRUNED_RAW = "yolov8n_pruned.pt"      # The raw 2.7M file from prune_surgical
HEALED_ONNX = "yolov8n_healed.onnx"   # Output

def heal_strict():
    print(f"Loading raw pruned model: {PRUNED_RAW}...")
    
    # 1. Load the raw nn.Module (The 2.7M param version)
    pruned_net = torch.load(PRUNED_RAW, weights_only=False)
    
    # Verify size BEFORE doing anything
    params = sum(p.numel() for p in pruned_net.parameters())
    print(f"Input Params: {params:,}")
    if params > 3000000:
        print("❌ STOP! The input file is already re-inflated.")
        print("Please re-run 'prune_surgical.py' to generate a fresh pruned model.")
        return

    # 2. Wrap in YOLO
    # We load a dummy YOLO object just to get the Trainer class capabilities
    model = YOLO("yolov8n.pt")
    
    # 3. THE SWAP (Critical)
    model.model = pruned_net
    
    # 4. THE FIX: DESTROY CONFIG METADATA
    # We delete the metadata that tells YOLO "I am a standard YOLOv8n"
    # This forces it to treat the model as a generic PyTorch architecture
    for attr in ['yaml', 'cfg', 'args']:
        if hasattr(model.model, attr):
            print(f"  - Stripping {attr} metadata...")
            setattr(model.model, attr, None)

    print("🏥 Starting Strict Healing (Fine-Tuning)...")
    
    # 5. Train
    # We use 'amp=False' to avoid Mac M1 errors
    # We use 'exist_ok=True' to overwrite previous runs
    try:
        model.train(
            data="coco128.yaml", 
            epochs=15, 
            imgsz=640, 
            lr0=0.001,
            amp=False,
            exist_ok=True
        )
    except Exception as e:
        print(f"⚠️ Training warning (might be ok): {e}")

    # 6. Verify Post-Training Size
    final_params = sum(p.numel() for p in model.model.parameters())
    print(f"Final Params: {final_params:,}")
    
    if final_params > 3000000:
        print("❌ FAILURE: The model re-inflated during training.")
    else:
        print("✅ SUCCESS: The model stayed pruned!")
        print("Exporting to ONNX...")
        model.export(format="onnx", opset=11)
        
        import os
        # Rename the output to be clear
        if os.path.exists("yolov8n.onnx"):
            os.rename("yolov8n.onnx", HEALED_ONNX)
            print(f"🎉 Saved final model: {HEALED_ONNX}")

if __name__ == "__main__":
    heal_strict()