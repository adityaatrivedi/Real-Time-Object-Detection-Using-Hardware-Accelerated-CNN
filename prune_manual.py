import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
PRUNING_RATIO = 0.4  # 40% pruning

def prune_manual():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    pytorch_model = model.model
    
    # Dummy input
    example_inputs = torch.randn(1, 3, 640, 640)
    
    # 1. FIND TARGET LAYERS
    print("Identifying internal bottleneck layers...")
    target_layers = []
    for m in pytorch_model.modules():
        if isinstance(m, Bottleneck):
            # Access the raw Conv2d layer inside the wrapper
            if hasattr(m.cv1, 'conv'):
                target_layers.append(m.cv1.conv)

    print(f"Found {len(target_layers)} internal layers to prune.")

    # 2. BUILD DEPENDENCY GRAPH
    print("Building Dependency Graph...")
    DG = tp.DependencyGraph()
    DG.build_dependency(pytorch_model, example_inputs=example_inputs)

    base_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Base Params: {base_params}")

    # 3. EXECUTE PRUNING
    print(f"Pruning internal layers with ratio {PRUNING_RATIO}...")
    
    count_pruned = 0
    
    for layer in target_layers:
        # Get current channels
        n_channels = layer.out_channels
        
        # Calculate how many to remove
        n_pruned = int(n_channels * PRUNING_RATIO)
        
        # --- FPGA OPTIMIZATION (Multiple of 8) ---
        # We want the REMAINING channels to be a multiple of 8
        target_channels = n_channels - n_pruned
        remainder = target_channels % 8
        
        if remainder != 0:
            # Prune a few more to snap to the nearest multiple of 8
            # e.g., Remainder 39 -> remove 7 more -> 32 remaining
            n_pruned += remainder
            
        # Safety: Don't kill the layer (keep at least 8 channels)
        if n_channels - n_pruned < 8:
            continue

        # --- MANUAL IMPORTANCE CALCULATION (The Fix) ---
        # Instead of using 'imp()', we just calculate L1 norm ourselves.
        # It's faster and cleaner.
        # 1. Flatten weights to [Out_Channels, Input*Kernel*Kernel]
        # 2. Calculate Sum of Absolute Values (L1 Norm) for each filter
        scores = torch.norm(layer.weight.view(n_channels, -1), p=1, dim=1)
        
        # 3. Find the indices of the 'n_pruned' smallest scores
        pruning_idxs = torch.argsort(scores)[:n_pruned].tolist()
        # -----------------------------------------------

        # Get the Group from the graph
        # This function finds all coupled layers that must also change
        group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=pruning_idxs)
        
        # Execute
        if group.has_dep: # Check if dependency graph is valid
            group.prune()
            count_pruned += 1

    # 4. VERIFICATION
    new_params = sum(p.numel() for p in pytorch_model.parameters())
    diff = base_params - new_params
    
    print(f"Pruned Params: {new_params}")
    
    if diff > 0:
        print(f"✅ SUCCESS! Modified {count_pruned} blocks.")
        print(f"Removed {diff} parameters ({(diff/base_params)*100:.2f}%)")
        print(f"New Model Size: {new_params / 1e6:.2f} M params")
        
        # Save
        torch.save(pytorch_model, SAVE_PATH)
        print(f"Saved to {SAVE_PATH}")
    else:
        print("❌ Failed to prune any parameters.")

if __name__ == "__main__":
    prune_manual()