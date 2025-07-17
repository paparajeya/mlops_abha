from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

iris_labels = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"])
            ]
            pred = model.predict([features])[0]
            prediction = iris_labels[pred]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
