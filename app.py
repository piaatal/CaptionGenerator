from flask import Flask, request, jsonify
import os
from transformers import pipeline
from PIL import Image, ImageFile
from flask_cors import CORS


# Allow PIL to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__) #flask application instance
CORS(app) #allows for requests from the frontend

# Load the image captioning model from hugging face, creating a pre-trained model pipeline
model_name = "Salesforce/blip-image-captioning-large"
classifier = pipeline(task="image-to-text", model=model_name)

#print("Model loaded successfully!")

#defines uploads folder to store images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#API endpoint that accepts POST requests
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # Check if the image file is provided in the request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        
        #saves the uploaded images to uploads directory
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        
        # Try to open and process the image using PIL
        image = Image.open(file_path).convert("RGB")
        
        # Generate caption using the image captioning model
        caption = classifier(image)
        
        #returning caption as a JSON response
        return jsonify({"caption": caption[0]['generated_text']})
    
    except Exception as e:
        # Catch any errors and return them as a response
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)

