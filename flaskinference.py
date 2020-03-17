import cv2
import pickle
import argparse
import importlib
from transformer import PoseExtractor

import pickle
import numpy as np
from flask import Flask, request

parser = argparse.ArgumentParser(description='Run inference on webcam video')
parser.add_argument('--config', type=str, default='conf',
                        help="name of config .py file inside config/ directory, default: 'conf'")
args = parser.parse_args()
config = importlib.import_module('config.' + args.config)

model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    model = pickle.load(open(config.classifier_model, 'rb'))


@app.route('/')
def home_endpoint():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)


