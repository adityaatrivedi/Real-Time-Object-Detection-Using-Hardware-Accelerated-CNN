import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, C2f, Bottleneck, Conv

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
PRUNING_RATIO = 0.3  # Remove 30% of channels

def prune_final():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    pytorch_model = model.model
    
    # 1. Setup Strategy
    example_inputs = torch.randn(1, 3, 640, 640)
    imp = tp.importance.MagnitudeImportance(p=1)

    # 2. CRITICAL FIX: Correctly Identify Layers to Ignore
    # We must explicitly find the 'Detect' head and ignore it.
    ignored_layers = []
    
    print("Analyzing layers to ignore...")
    for m in pytorch_model.modules():
        # Ignore the final Detect head (it has hardcoded channel requirements)
        if isinstance(m, Detect):
            print(f" - Found Detect Head: Ignoring {m}")
            ignored_layers.append(m)
            
    # 3. Setup Pruner
    print("Building Dependency Graph (this might take a moment)...")
    pruner = tp.pruner.MagnitudePruner(
        pytorch_model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=PRUNING_RATIO,
        ignored_layers=ignored_layers,
        
        # FPGA OPTIMIZATION:
        # Force channel counts to be multiples of 8.
        # This prevents the pruner from creating weird layer sizes (e.g., 31 channels)
        # that your FPGA hardware logic handles poorly.
        round_to=8, 
    )

    base_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Base Params: {base_params}")

    # 4. Execute
    print(f"Pruning with ratio {PRUNING_RATIO}...")
    pruner.step()

    # 5. Verification
    new_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Pruned Params: {new_params}")
    
    # Check if we actually removed something
    diff = base_params - new_params
    if diff == 0:
        print("\n⚠️  WARNING: Pruning still did nothing.")
        print("Try increasing PRUNING_RATIO to 0.5 or checking if 'Detect' is imported correctly.")
        raise RuntimeError("Pruning failed to remove any weights.")
        
    print(f"✅ SUCCESS! Removed {diff} parameters ({(diff/base_params)*100:.2f}%)")
    print(f"New Model Size: {new_params / 1e6:.2f} M params")

    # 6. Save
    # Save the entire object so we can load it back easily
    torch.save(pytorch_model, SAVE_PATH)
    print(f"Saved pruned model to {SAVE_PATH}")
    
    # 7. Quick Verify Run
    print("Verifying pruned model forward pass...")
    with torch.no_grad():
        pytorch_model(example_inputs)
    print("✅ Model runs correctly!")

if __name__ == "__main__":
    prune_final()