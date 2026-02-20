import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck

# --- CONFIG ---
MODEL_PATH = "yolov8n.pt"
SAVE_PATH = "yolov8n_pruned.pt"
PRUNING_RATIO = 0.4  # Remove 40% of internal channels

def prune_surgical():
    print(f"Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    pytorch_model = model.model
    
    # Dummy input for tracing a single bottleneck
    # (Bottlenecks usually take C_in channels, but we can just use a generic size)
    # The tracer mainly needs the shape flow, not exact values.
    
    print("Starting Surgical Pruning...")
    count_modified = 0
    total_params_removed = 0
    
    # 1. Iterate through every module in the model
    for name, module in pytorch_model.named_modules():
        
        # 2. We only target the 'Bottleneck' blocks (the building blocks of YOLO)
        if isinstance(module, Bottleneck):
            # The Bottleneck has an internal Conv (cv1) and an exit Conv (cv2).
            # We want to prune cv1's output (which is cv2's input).
            # This is "Hidden" inside the block, so it's safe!
            
            target_layer = module.cv1.conv
            n_channels = target_layer.out_channels
            
            # 3. Calculate Pruning Amount (Multiple of 8 for FPGA)
            n_pruned = int(n_channels * PRUNING_RATIO)
            remainder = (n_channels - n_pruned) % 8
            if remainder != 0:
                n_pruned += remainder # Snap to nearest multiple of 8
                
            # Safety Check: Don't kill the layer
            if (n_channels - n_pruned) < 8:
                continue
                
            # 4. Build a TINY Graph just for this module
            # We treat this one bottleneck as if it were the whole world.
            # We need to know the input shape to trace it.
            # Bottleneck inputs usually match the previous layer's output.
            # We can guess a safe shape or capture it via a hook. 
            # TRICK: We don't actually need to run the forward pass if we manually
            # prune the weight tensor, but using the pruner is safer.
            
            # Manual Pruning (The most robust way for this specific case)
            # Since we know exactly what a Bottleneck is: 
            # cv1.weight is [Out, In, k, k]
            # cv2.weight is [Out, In, k, k]
            
            # We calculate L1 norm for cv1 output filters
            importance = torch.norm(target_layer.weight.view(n_channels, -1), p=1, dim=1)
            pruning_idxs = torch.argsort(importance)[:n_pruned].tolist()
            
            # --- SURGERY TIME ---
            # 1. Prune cv1 OUTPUT (Rows)
            tp.prune_conv_out_channels(target_layer, pruning_idxs)
            
            # 2. Prune cv2 INPUT (Columns) - This must match cv1's output!
            # module.cv2.conv is the next layer
            tp.prune_conv_in_channels(module.cv2.conv, pruning_idxs)
            
            # 3. Update Batch Norms (if they exist)
            if hasattr(module.cv1, 'bn'):
                tp.prune_batchnorm_out_channels(module.cv1.bn, pruning_idxs)
                
            count_modified += 1
            total_params_removed += n_pruned * (target_layer.kernel_size[0]**2 * target_layer.in_channels) # Approx
            
    # 5. Final Report
    base_params = 3157200 # Known size of YOLOv8n
    new_params = sum(p.numel() for p in pytorch_model.parameters())
    
    print(f"✅ SURGERY COMPLETE!")
    print(f"Modified {count_modified} Bottleneck blocks.")
    print(f"Original Params: {base_params}")
    print(f"New Params:      {new_params}")
    print(f"Removed:         {base_params - new_params}")
    
    torch.save(pytorch_model, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    prune_surgical()