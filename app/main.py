# app/app.py
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import predict_image

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", prediction="No file uploaded.")
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", prediction="Empty filename.")
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)
            prediction = predict_image(image_path)
    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)