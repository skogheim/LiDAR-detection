import cv2
from matplotlib import pyplot as plt

def draw_gt_boxes(image_path, label_path):
    """
    Draw ground truth bounding boxes on an image.

    This function reads an image and corresponding label file, then draws rectangles around each object as specified
    in the label file. Each object's class name is displayed above its bounding box.

    Parameters:
    - image_path (str): Path to the image file.
    - label_path (str): Path to the label file containing bounding box coordinates and class IDs.

    The label file should contain one bounding box per line, each line formatted as:
    class_id centerX_norm centerY_norm width_norm height_norm
    where coordinates are normalized between 0 and 1 relative to the image dimensions.

    Dependencies:
    - OpenCV: Used for image manipulation.
    - Matplotlib: Used to display the image.
    """

    # Define class names based on your data
    class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider']
    
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return
    
    # Extract dimensions of the image
    height, width, _ = image.shape
    
    # Open and read the label file
    with open(label_path, 'r') as file:
        boxes = file.readlines()
    
    # Convert image color space from BGR (OpenCV default) to RGB for displaying
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Iterate over each bounding box in the label file
    for box in boxes:
        parts = box.strip().split()
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        
        # Calculate pixel coordinates of the bounding box from normalized values
        x = int((cx - bw / 2) * width)
        y = int((cy - bh / 2) * height)
        w = int(bw * width)
        h = int(bh * height)
        
        # Draw the bounding box on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red color, 2px thickness
        # Annotate the image with class name above the bounding box
        cv2.putText(image, class_names[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Use Matplotlib to display the image
    plt.imshow(image)
    plt.title('Image with GT Boxes')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

# Example usage
image_path = 'NAPLab-LiDAR-Dataset/images/test/frame_001493.PNG'  # Replace with your image file path
label_path = 'NAPLab-LiDAR-Dataset/labels/test/frame_001493.txt'  # Replace with your label file path

draw_gt_boxes(image_path, label_path)
