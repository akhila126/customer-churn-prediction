
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/churn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    senior = int(request.form["SeniorCitizen"])
    partner = int(request.form["Partner"])
    dependents = int(request.form["Dependents"])
    tenure = int(request.form["tenure"])
    monthly = float(request.form["MonthlyCharges"])
    total = float(request.form["TotalCharges"])

    features = np.array([[gender,
                          senior,
                          partner,
                          dependents,
                          tenure,
                          monthly,
                          total]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Customer is likely to Churn."
    else:
        result = "Customer is likely to Stay."

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)