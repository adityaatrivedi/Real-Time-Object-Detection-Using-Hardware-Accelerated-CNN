from ultralytics import YOLO

def heal_correctly():
    # Load the formatted custom model directly
    # This forces YOLO to accept the architecture AS IS
    print("Loading custom pruned model...")
    model = YOLO("yolov8n_pruned_formatted.pt")
    
    print("🏥 Starting Fine-Tuning (Healing)...")
    
    # Train
    # We use a very low LR to nudge the weights back into alignment
    model.train(
        data="coco128.yaml", 
        epochs=15, 
        imgsz=640, 
        lr0=0.001,
        amp=False # Disable Mixed Precision to avoid the "Half" error on Mac
    )
    
    print("✅ Healing Complete.")
    
    # Export
    print("Exporting to ONNX...")
    model.export(format="onnx", opset=11)
    
    import os
    if os.path.exists("yolov8n_pruned_formatted.onnx"):
        os.rename("yolov8n_pruned_formatted.onnx", "yolov8n_final_pruned.onnx")
        print("🎉 Final Model: yolov8n_final_pruned.onnx")

if __name__ == "__main__":
    heal_correctly()