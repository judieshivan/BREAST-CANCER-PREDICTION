import os
import pathlib

import requests
from flask import Flask, session, abort, redirect, request, render_template
import math
import pickle
import numpy as np
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests

#
app = Flask("Breast Cancer Prediction")




@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")



#
# @app.route('/prediction_', methods=['POST', 'GET'])
# def prediction_():
#
#     print([x for x in request.form.values()])
#     print(len([x for x in request.form.values()]))
#
#
#     # int_features = [float(x) for x in request.form.values()]
#     int_features=[1.0, 0.0, 1.0, 87.0, 7.0, 0.0, 1.0, 1.0, 4.0, 87.0, 0.0, 67.0, 1.0, 1.0, 34.0, 1.0, 5.0, 7.0, 1.0, 0.0, 6.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
#
#     print(int_features)
#
#
#     array_numpy_features = [np.array(int_features)]
#     prediction = linear_logistic_model.predict(array_numpy_features)
#     print(prediction, "<== predictions")
#     output = prediction[0]
#
#
#     return render_template(
#         'pred.html', pred='Attack Type Results : {}'.format(output),
#         bhai="Please fill in for more!",
#         )
#
#
#



if __name__ == "__main__":
    app.run(debug=True)
