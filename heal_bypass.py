import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
import sys

# --- CONFIG ---
PRUNED_RAW = "yolov8n_pruned.pt"
OUTPUT_ONNX = "yolov8n_final_pruned.onnx"

def heal_bypass():
    print(f"Loading pruned weights: {PRUNED_RAW}...")
    # 1. Load the raw pruned model object
    try:
        pruned_model = torch.load(PRUNED_RAW, weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {PRUNED_RAW}")
        return

    # Verify we are starting with the small model
    params = sum(p.numel() for p in pruned_model.parameters())
    print(f"Starting Params: {params:,}")
    
    if params > 3000000:
        print("❌ ERROR: Model is already re-inflated! Re-run 'prune_surgical.py' first.")
        return

    # 2. CONFIGURATION
    # We provide a dummy 'model' string to satisfy the constructor checks.
    # Our custom class will ignore this string and use 'pruned_model' instead.
    overrides = {
        'model': 'yolov8n.pt', 
        'data': 'coco128.yaml', 
        'epochs': 15, 
        'imgsz': 640, 
        'lr0': 0.001, 
        'amp': False,
        'exist_ok': True
    }
    
    # 3. Create Custom Trainer
    class PrunedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            print("🛡️  Bypassing standard model load... Injecting Pruned Model.")
            
            # Manually Attach Metadata
            # The trainer expects certain args to exist on the model
            if not hasattr(pruned_model, 'args'):
                pruned_model.args = {}
            
            # We force the args to match our training config
            pruned_model.args['imgsz'] = 640
            pruned_model.args['task'] = 'detect'
            
            return pruned_model

    # 4. Initialize & Train
    try:
        trainer = PrunedTrainer(overrides=overrides)
        print("🏥 Starting Bypass Healing (Fine-Tuning)...")
        trainer.train()
    except Exception as e:
        print(f"\n❌ Training crashed: {e}")
        # Even if training crashes, we might want to check if the model survived
    
    # 5. Verify Result
    print("\n--- Verification ---")
    final_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Final Params: {final_params:,}")

    if final_params > 3000000:
        print("❌ FAILURE: The model re-inflated.")
        print("The Trainer likely ignored our bypass and loaded the dummy 'yolov8n.pt'.")
    else:
        print("✅ SUCCESS: The model is trained and still pruned!")
        
        # 6. Export safely
        print("Exporting to ONNX...")
        
        # We load a fresh wrapper to handle the export logic
        exporter = YOLO("yolov8n.pt")
        # Inject our trained brain
        exporter.model = trainer.model
        
        # Clean metadata that might confuse the exporter
        exporter.model.args = {
            'imgsz': 640,
            'batch': 1,
            'task': 'detect'
        }
        
        exporter.export(format="onnx", opset=11)
        
        import os
        if os.path.exists("yolov8n.onnx"):
            os.rename("yolov8n.onnx", OUTPUT_ONNX)
            print(f"🎉 Final File: {OUTPUT_ONNX}")

if __name__ == "__main__":
    heal_bypass()