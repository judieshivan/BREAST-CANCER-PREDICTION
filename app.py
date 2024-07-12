import os
import pathlib
import joblib
import os
import pathlib

import mysql.connector
import requests
from flask import Flask, session, abort, redirect, request, render_template
import numpy as np
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests




app = Flask("Breast Cancer Prediction")
app.secret_key = "Breast cancer Prediction"

scaler = joblib.load('./models/scaler.pkl')  # The scaler fitted on the training data
pca = joblib.load('./models/pca.pkl')  # The PCA fitted on the training data
logreg = joblib.load('./models/HybridModel.pkl')  # The trained RandomForestClassifier




os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

GOOGLE_CLIENT_ID = "457054239703-ej0d9ds3qufaa3e88easu73vnj0r8bvl.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:5000/callback"
)


def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session:
            return abort(401)  # Authorization required
        else:
            return function()

    return wrapper


@app.route("/login")
def login():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)  # State does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    session["picture"] = id_info.get("picture")  # Get the profile image URL
    return redirect("/home")




@app.route("/logout")
def logout():
    session.clear()
    return render_template("login.html")


@app.route("/")
def index():
    return render_template("login.html")


@app.route("/home")
@login_is_required
def home():
    return render_template("index.html", User='Hi, {}'.format(session['name']))



@app.route("/pred1")
def pred1():
    return render_template("pred1.html", User='Hi, {}'.format(session['name']))


@app.route("/contact")
def contact():
    return render_template("contact.html", User='Hi, {}'.format(session['name']))


@app.route("/history")
def history():
    # Establish the connection
    connection = mysql.connector.connect(
        host='localhost',  # Hostname of the MySQL server
        user='root',  # Your MySQL username
        database='cancer_db'  # Name of the database you want to connect to
    )

    # Create a cursor object
    cursor = connection.cursor()

    # Execute a query (corrected to use proper string formatting)
    user_email = session['name']
    query = f"SELECT * FROM user_data WHERE user_email = '{user_email}'"
    cursor.execute(query)
    # Fetch the results
    results = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return render_template("History.html", results=results, User='Hi, {}'.format(session['name']))



@app.route("/pred1_", methods=['POST', 'GET'])
def pred1_():
    int_features = [x for x in request.form.values()]
    if not int_features or "" in int_features:
        return render_template("pred1.html", prediction="Please enter all required features.", User='Hi, {}'.format(session['name']))

    int_features = [float(x) for x in int_features]
    print(int_features)
    array_numpy_features = np.array(int_features).reshape(1, -1)

    # Preprocess the input features
    X_scaled = scaler.transform(array_numpy_features)  # Scale the input features
    X_pca = pca.transform(X_scaled)  # Apply PCA transformation

    # Predict using the trained classifier
    y_pred = logreg.predict(X_pca)
    print("Predicted values for new user input data:", y_pred)
    result_str=y_pred[0]

    save_prediction_to_db(str(session['name']),int_features, str(result_str))

    return render_template("pred1.html", prediction=f'Your predicted value is : {y_pred[0]}',User='Hi, {}'.format(session['name']))

    

def save_prediction_to_db(user_email,features, prediction):
    # Establish the connection
    connection = mysql.connector.connect(
        host='localhost',  # Hostname of the MySQL server
        user='root',  # Your MySQL username
        database='cancer_db'  # Name of the database you want to connect to
    )

    # Create a cursor object
    cursor = connection.cursor()

    # Insert the prediction into the database
    query = """
    INSERT INTO user_data (radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst, result,user_email)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)
    """
    cursor.execute(query, (*features, prediction,str(user_email),))

    # Commit the transaction
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()





if __name__ == "__main__":
    app.run(debug=True)
