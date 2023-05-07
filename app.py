import json
import helpers
import numpy as np
import pandas as pd
from flask import jsonify
from flask import Flask, request, render_template

#pip install -r requirements.txt
# To install all packages
# Install xgboost from conda
# pip install -r requirements.txt

#Create an app object using the Flask class. 
app = Flask(__name__)
app.static_folder = 'static'


# Load dictionary with brands and their modules
with open("brands_models.json", "r") as file:
    brands_models = json.load(file)

brands  = list(brands_models.keys())

@app.route('/')
def home():
    return render_template('index.html', len = len(brands), brands = brands)


@app.route('/predict',methods=['POST'])
def predict():
    # Important to install scikit-learn v 1.2.1

    # Extract features from the request object 
    car = helpers.extract_features(request)

    # Perform basic preprocessing on the car object 
    car = helpers.basic_preprocessing(car)

    # Create a copy of the preprocessed car object to use with the TensorFlow model
    car_tf = car.copy()

    # Perform decision tree model-specific preprocessing on the car object 
    car = helpers.dt_model_preprocessing(car)

    # Use the decision tree model to make a prediction
    pred_dt = helpers.dt_model_prediction(car)

    # Use the TensorFlow model to make a prediction 
    pred_tf = helpers.tf_model_prediction(car_tf)


    return render_template(
        'index.html', 
        len = len(brands),
        brands = brands, 
        prediction_text='DT Price is '+ str(pred_dt)+ ' Tensorflow: ' + str(pred_tf))


@app.route("/get_brands/<brand>", methods=["GET"])
def get_brands(brand):
    models = brands_models[brand]
    # do something with the selected option
    return jsonify(models=models)  

if __name__ == "__main__":
    app.run(debug=True)