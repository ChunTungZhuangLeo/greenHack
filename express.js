const express = require("express");
const multer = require("multer");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = 3000;

// Serve static files (frontend)
app.use(express.static("public"));

// Configure Multer for file upload
const storage = multer.diskStorage({
    destination: "uploads/",
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});
const upload = multer({ storage: storage });

// API endpoint to upload an image and process with Roboflow
app.post("/upload", upload.single("image"), async (req, res) => {
    try {
        const imagePath = path.join(__dirname, "uploads", req.file.filename);
        const imageBase64 = fs.readFileSync(imagePath, { encoding: "base64" });

        // Send image to Roboflow API
        const response = await axios({
            method: "POST",
            url: "https://outline.roboflow.com/oil-spill-segmentation/3",
            params: {
                api_key: "GCoTo1qoR1y3Wu5K3NpS"
            },
            data: imageBase64,
            headers: {
                "Content-Type": "application/x-www-form-urlencoded"
            }
        });

        // Save the result image from Roboflow
        const outputImagePath = path.join(__dirname, "public", "output.png");
        fs.writeFileSync(outputImagePath, Buffer.from(response.data.image, "base64"));

        res.json({ success: true, imageUrl: "/output.png" });

    } catch (error) {
        console.error(error);
        res.status(500).json({ success: false, message: "Error processing image" });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
