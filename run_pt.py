from ultralytics import YOLO
import cv2

# 1. Load the model (use 'yolov8n.pt', 'yolov8s.pt', or your custom trained 'best.pt')
model = YOLO("yolov3tiny_facemask.pt")  
# 2. Run Inference
# source can be a file path, a URL, or '0' for webcam
print("Running detection...")
results = model("https://ultralytics.com/images/bus.jpg")

# 3. Work with Results
for result in results:
    # Print boxes to console (x, y, w, h)
    for box in result.boxes:
        print(f"Detected class {int(box.cls)} with conf {float(box.conf):.2f}")
    
    # Show the image on screen
    result.show()  
    
    # Save the image to disk
    result.save(filename="output_result.jpg")

print("Done! Check 'output_result.jpg'")