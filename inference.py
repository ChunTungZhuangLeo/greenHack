import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="GCoTo1qoR1y3Wu5K3NpS"
)

# Path to your image
image_path = r"C:\Users\z2750\Desktop\greenHack\129790834_3855777544455850_2629439794151397678_n_jpg.rf.0ba829289bb2530f0d361ad2512e1c88.jpg"

# Perform inference
result = CLIENT.infer(image_path, model_id="oil-spill-segmentation/3")

# Read the input image
image = cv2.imread(image_path)

# Ensure predictions exist
if "predictions" in result:
    for obj in result["predictions"]:
        points = obj["points"]  # Extract polygon points
        class_id = obj["class_id"]  # Class ID (optional)

        # Convert points to numpy array
        polygon = np.array([[p["x"], p["y"]] for p in points], np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Draw filled polygon (segmentation mask)
        cv2.fillPoly(image, [polygon], (0, 255, 0))  # Green mask

# Save and show the output image
output_path = r"C:\Users\z2750\Desktop\greenHack\save_test.jpeg"
cv2.imwrite(output_path, image)
cv2.imshow("Segmented Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Segmented image saved as: {output_path}")
