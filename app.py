from flask import Flask, redirect, url_for, request, render_template

import tensorflow as tf
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops

ops.reset_default_graph()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

import numpy as np
import h5py
import PIL
from PIL import Image
import os

app = Flask(__name__)


MODEL_ARCHITECTURE = "./models/BrainTumorWeightsJson.json"
MODEL_WEIGHTS = "./models/BrainTumorWeights.h5"

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print("Model loaded. Check http://127.0.0.1:5000/")


# Flask Routes
@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about-us")
def aboutus():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def upload():
    if request.method == "POST":

    
        f = request.files["asfile"]

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, "uploads",
                                 secure_filename(f.filename))
        f.save(file_path)

        
        prediction = model_predict(file_path, model)

        Answer = ""
        if (prediction < 1):
            Answer = "Brain is Normal"
        else:
            Answer = "Brain has Tumor"

        return render_template("index.html", Answer=Answer)


# MODEL FUNCTIONS
def model_predict(img_path, model):

    print(model)

    MRIScan = image.load_img(img_path, target_size=(150, 150))
    MRIScan = image.img_to_array(MRIScan)
    MRIScan = np.expand_dims(MRIScan, axis=0)
    MRIScan = MRIScan / 255
    print(MRIScan.shape)

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate= 0.002, epsilon=1e-03),
        metrics=["accuracy"],
    )

    prediction = model.predict_classes(MRIScan)
    print("Prediction Class: ", prediction)

    return prediction


if __name__ == "__main__":
    app.run(debug=True)
