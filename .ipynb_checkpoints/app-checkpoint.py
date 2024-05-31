import os
import pathlib

import numpy as np
import requests
from flask import Flask, session, abort, redirect, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle


app = Flask("Breast Cancer Prediction")

# Ensure to load the trained models and preprocessing objects
scaler = joblib.load('./models/scaler.pkl')  # The scaler fitted on the training data
pca = joblib.load('./models/pca.pkl')  # The PCA fitted on the training data
logreg = joblib.load('./models/logreg.pkl')  # The trained RandomForestClassifier

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/pred1")
def pred1():
    return render_template("pred1.html")

@app.route("/pred2")
def pred2():
    return render_template("pred2.html")


@app.route("/pred3")
def pred3():
    return render_template("pred3.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/pred1_", methods=['POST', 'GET'])
def pred1_():
    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    array_numpy_features = np.array(int_features).reshape(1, -1)

    # Preprocess the input features
    X_scaled = scaler.transform(array_numpy_features)  # Scale the input features
    X_pca = pca.transform(X_scaled)  # Apply PCA transformation

    # Predict using the trained classifier
    y_pred = logreg.predict(X_pca)
    print("Predicted values for new user input data:", y_pred)

    return render_template("pred1.html")







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
