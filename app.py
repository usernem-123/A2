from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/results"

# Load your trained model only once (fast)
model = YOLO("runs/detect/train/weights/best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        # Save uploaded image
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

        # Run YOLO inference
        results = model(
            img_path,
            save=True,
            project="static/results",
            name="preds",
            exist_ok=True
        )

        # Count persons
        count = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    count += 1

        output_image = "static/results/preds/" + file.filename

        return render_template("index.html", output_image=output_image, count=count)

    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
