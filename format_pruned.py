import torch

# --- CONFIG ---
INPUT_RAW = "yolov8n_pruned.pt"
OUTPUT_FORMATTED = "yolov8n_pruned_formatted.pt"

def format_checkpoint():
    print(f"Loading raw pruned model: {INPUT_RAW}...")
    # Load the raw nn.Module
    model = torch.load(INPUT_RAW, weights_only=False)
    
    # Verify it is actually pruned
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameter Count: {params:,}")
    
    if params > 3000000:
        print("❌ WARNING: This model looks like the full unpruned version!")
        print("Did you overwrite it?")
        return

    # Create a fake checkpoint dictionary that Ultralytics expects
    ckpt = {
        'model': model,
        'epoch': -1,
        'best_fitness': None,
        'optimizer': None,
    }
    
    # Save it
    torch.save(ckpt, OUTPUT_FORMATTED)
    print(f"✅ Saved formatted checkpoint: {OUTPUT_FORMATTED}")
    print("You can now load this directly into YOLO() without it resetting.")

if __name__ == "__main__":
    format_checkpoint()