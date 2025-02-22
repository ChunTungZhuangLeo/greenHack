import os
import cv2
import numpy as np
import base64
from flask import Flask, request, render_template, jsonify
from inference_sdk import InferenceHTTPClient

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="GCoTo1qoR1y3Wu5K3NpS"
)

# Flask Route: Homepage
@app.route("/")
def index():
    return render_template("index.html")

# Flask Route: Handle Image Upload & Segmentation
@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Perform inference
    result = CLIENT.infer(image_path, model_id="oil-spill-segmentation/3")

    # Read input image
    image = cv2.imread(image_path)

    # Ensure predictions exist
    if "predictions" in result:
        for obj in result["predictions"]:
            points = obj["points"]  # Extract polygon points

            # Convert points to numpy array
            polygon = np.array([[p["x"], p["y"]] for p in points], np.int32)
            polygon = polygon.reshape((-1, 1, 2))

            # Draw filled polygon (segmentation mask)
            cv2.fillPoly(image, [polygon], (0, 255, 0))  # Green mask

    # Save segmented output image
    output_filename = "segmented_" + file.filename
    output_path = os.path.join(RESULT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    # Convert output image to base64 for display in frontend
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": image_base64, "output_path": output_path})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
