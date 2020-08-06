# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open("./model/classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Hello World!"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():

    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """

    variance = int(request.args.get('variance'))
    skewness = int(request.args.get('skewness'))
    curtosis = int(request.args.get('curtosis'))
    entropy = int(request.args.get('entropy'))

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted values is " + str(prediction)
    # print(prediction)
    # return prediction


@app.route('/predict_file', methods=["POST"])
def predict_note_file():

    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values
    """

    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return str(list(prediction))


# http://127.0.0.1:5000/apidocs/
if __name__ == '__main__':
    app.run()
