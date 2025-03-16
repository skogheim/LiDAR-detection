"""
This script is adapted from the Ultralytics YOLO documentation available at:
https://docs.ultralytics.com/modes/predict/#plotting-results

The code is used under the terms of the specific license provided by Ultralytics, designed to demonstrate
how to load a pretrained YOLOv8 model, perform inference, and visualize the results.
"""

from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8s.pt')

# Run inference on given picture
results = model(['NAPLab-LiDAR-Dataset/images/test/frame_000567.PNG'])  # results list of results objects

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f'results{i}.jpg')
    
    
