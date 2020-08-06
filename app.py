# -*- coding: utf-8 -*-

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
pickle_in = open("./model/classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def welcome():
    return "Hello World!"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():
    variance = int(request.args.get('variance'))
    skewness = int(request.args.get('skewness'))
    curtosis = int(request.args.get('curtosis'))
    entropy = int(request.args.get('entropy'))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted values is " + str(prediction)


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted values for the csv is " + str(list(prediction))


if __name__ == '__main__':
    app.run()
