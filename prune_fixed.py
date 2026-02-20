import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
PRUNING_RATIO = 0.4  # 40% pruning

def prune_fixed():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    pytorch_model = model.model
    
    # Dummy input for graph tracing
    example_inputs = torch.randn(1, 3, 640, 640)
    
    # Importance Strategy
    imp = tp.importance.MagnitudeImportance(p=1)

    print("Identifying internal bottleneck layers...")
    
    # 1. FIND TARGET LAYERS
    target_layers = []
    for m in pytorch_model.modules():
        if isinstance(m, Bottleneck):
            # Access the raw Conv2d layer inside the wrapper
            # m.cv1 is the wrapper (Ultralytics 'Conv')
            # m.cv1.conv is the actual PyTorch 'Conv2d'
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
        
        # FPGA SAFETY CHECK:
        # We want the remaining channels to be multiples of 8 if possible.
        # Current: 64 -> Prune 25 -> Remainder 39 -> Bad for FPGA.
        # Goal:    64 -> Prune 24 -> Remainder 40 -> Good.
        remainder = n_channels - n_pruned
        if remainder % 8 != 0:
            adjustment = remainder % 8
            n_pruned += adjustment # Remove a few more to make remainder multiple of 8
        
        # Safety: Don't prune if we remove everything
        if n_channels - n_pruned < 8:
            continue

        # Get the indices of the least important filters
        pruning_idxs = imp(layer, idxs=None) # Get all scores
        # Sort and pick the smallest 'n_pruned' indices
        pruning_idxs = torch.argsort(pruning_idxs)[:n_pruned].tolist()
        
        # Get the plan from the graph
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv_out_channels, idxs=pruning_idxs)
        
        # Execute
        if pruning_plan is not None:
            pruning_plan.exec()
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
    prune_fixed()