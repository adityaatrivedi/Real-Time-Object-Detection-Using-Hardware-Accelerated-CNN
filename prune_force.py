import torch
import torch_pruning as tp
from ultralytics import YOLO

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
RATIO = 0.3  # Prune 30%

def prune_force():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH) 
    pytorch_model = model.model
    
    # 1. Setup Strategy
    example_inputs = torch.randn(1, 3, 640, 640)
    
    # Importance: Rank by L1 Norm (magnitude)
    imp = tp.importance.MagnitudeImportance(p=1)

    # 2. Setup Pruner
    # We ignore the absolute final layer (Detect) to prevent shape mismatches
    ignored_layers = []
    for m in pytorch_model.modules():
        if isinstance(m, torch.nn.modules.conv.Conv2d) and m.out_channels == 80:
            ignored_layers.append(m)

    print(f"Base Params: {sum(p.numel() for p in pytorch_model.parameters())}")

    pruner = tp.pruner.MagnitudePruner(
        pytorch_model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        
        # --- THE FIX IS HERE ---
        pruning_ratio=RATIO,  # Renamed from 'ch_sparsity'
        # -----------------------
        
        ignored_layers=ignored_layers,
    )

    # 3. Execute
    print(f"Pruning with ratio {RATIO}...")
    pruner.step()

    # 4. Verification
    new_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Pruned Params: {new_params}")
    
    if new_params >= 3157200:
        raise RuntimeError("❌ PRUNING FAILED! Parameter count did not drop.")
    
    print(f"✅ Success! Removed {3157200 - new_params} parameters.")

    # 5. Save the pure PyTorch model (safer than saving YOLO wrapper)
    torch.save(pytorch_model, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    prune_force()