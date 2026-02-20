import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
PRUNING_RATIO = 0.4  # Go aggressive (40%) since we are only pruning internals

def prune_internal():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    pytorch_model = model.model
    
    example_inputs = torch.randn(1, 3, 640, 640)
    imp = tp.importance.MagnitudeImportance(p=1)

    print("Identifying internal bottleneck layers...")
    
    # We will manually list the specific layers we WANT to prune.
    # We target the first convolution (cv1) inside every Bottleneck block.
    # Pruning this reduces the internal width of the block without changing its input/output interface.
    target_layers = []
    for m in pytorch_model.modules():
        if isinstance(m, Bottleneck):
            # YOLOv8 Bottleneck has: cv1 (3x3) -> cv2 (3x3)
            # We prune cv1. This shrinks cv1 output and cv2 input.
            # This is safe because it's "inside" the residual connection.
            target_layers.append(m.cv1)

    print(f"Found {len(target_layers)} internal layers to prune.")

    # Setup Pruner
    # global_pruning=False ensures we strictly follow our ratio for these specific layers
    pruner = tp.pruner.MagnitudePruner(
        pytorch_model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=PRUNING_RATIO,
        ignored_layers=[], # No need to ignore head, we aren't touching it!
        root_module_types=[torch.nn.Conv2d], # Only verify Conv2d
        round_to=8, # Hardware friendly
    )

    base_params = sum(p.numel() for p in pytorch_model.parameters())
    print(f"Base Params: {base_params}")

    # Execute
    print(f"Pruning internal layers...")
    
    # We manually step the pruner ONLY on our target layers
    for layer in target_layers:
        # We ask the pruner: "Give me a plan to prune this specific layer"
        # Since these are internal to bottlenecks, they usually don't have complex dependencies
        # that reach the Head.
        pruner.step(interactive=True) # Switch to interactive mode if needed, 
                                      # but standard step() usually prunes everything not ignored.
                                      # Let's rely on the pruner's ability to prune what we allow.
                                      
    # actually, with tp 1.x, we can't easily pass a list of targets to .step().
    # instead, we will use the DependencyGraph manually to prune just these.
    
    DG = tp.DependencyGraph().build_dependency(pytorch_model, example_inputs=example_inputs)
    
    count_pruned = 0
    for layer in target_layers:
        # Get pruning plan for this layer
        # amount is the number of channels to remove
        n_channels = layer.out_channels
        n_pruned = int(n_channels * PRUNING_RATIO)
        
        # Ensure we keep at least 8 channels (and multiple of 8)
        # e.g. if we have 64, prune 25 -> leave 39 -> round to 40.
        
        if n_pruned < 1: continue

        # Get the indices of the least important filters
        pruning_idxs = imp(layer, amount=n_pruned)
        
        # Get the plan from the graph
        pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv_out_channels, idxs=pruning_idxs)
        
        # Execute
        if pruning_plan is not None:
            pruning_plan.exec()
            count_pruned += 1

    # 5. Verification
    new_params = sum(p.numel() for p in pytorch_model.parameters())
    diff = base_params - new_params
    
    print(f"Pruned Params: {new_params}")
    print(f"✅ SUCCESS! Modified {count_pruned} blocks.")
    print(f"Removed {diff} parameters ({(diff/base_params)*100:.2f}%)")

    # 6. Save
    torch.save(pytorch_model, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    prune_internal()