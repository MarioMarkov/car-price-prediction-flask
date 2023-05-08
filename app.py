import json
import helpers
import numpy as np
import pandas as pd
from flask import jsonify
from flask import Flask, request, render_template, redirect, url_for
import os
#pip install -r requirements.txt
# To install all packages
# Install xgboost from conda
# pip install -r requirements.txt

# run npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch

#Create an app object using the Flask class. 
app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = 'secret'

# Load dictionary with brands and their modules
with open("brands_models.json", "r") as file:
    brands_models = json.load(file)

brands  = list(brands_models.keys())

@app.route('/',methods=['GET'])
def home():
    # if(hasattr(request.args['alert']))
    # alert = json.loads(request.args['alert'])  # counterpart for url_for()
    # if(alert){
    #     print(alert)
    # }
    if 'alert' in request.args and 'vote_model' in request.args:
        alert = request.args['alert']
        vote_model = request.args['vote_model']
        print(alert)
        return render_template('index.html', len = len(brands), brands = brands,
                               alert = alert,
                               vote_model = vote_model)
    
    # Render the template with empty form fields
    return render_template('index.html', len = len(brands), brands = brands)

@app.route('/prediction',methods=['POST'])
def vote():
    vote_model = request.form['model']
    return redirect(url_for('.home',alert = "success", vote_model=vote_model))


@app.route('/',methods=['POST'])
def predict():
    # Important to install scikit-learn v 1.2.1
    # Retrieve the values of the submitted form fields
    brand = request.form['brand']
    model = request.form['model']
    year = request.form['year']
    fuel = request.form['fuel']
    kms = request.form['kms']
    transmission = request.form['transmission']
    door_2 = request.form['2door']
    color = request.form['color']
    type_car = request.form['type']
    displacement = request.form['displacement']
    hp = request.form['hp']
    #euro = request.form['euro']
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

    results = json.dumps({"pred_dt":pred_dt,"pred_tf":pred_tf})
    #session['results'] = results
    # return render_template(
    #     'index.html', 
    #     len = len(brands),
    #     brands = brands, 
    #     prediction_text='DT Price is '+ str(pred_dt)+ ' Tensorflow: ' + str(pred_tf),
    #     brand = brand, model =model, year =year, fuel =fuel, kms =kms, transmission =transmission, door_2 = door_2, color =color, type_car = type_car, displacement =displacement, hp =hp)
    #return redirect("/prediction",pred_dt= pred_dt , pred_tf= pred_tf)
    return redirect(url_for('.display_prediction', results=results))

    #return render_template('prediction.html',pred_dt= pred_dt , pred_tf= pred_tf)


@app.route("/prediction", methods=["GET"])
def display_prediction():
    results = json.loads(request.args['results'])  # counterpart for url_for()
    #results = session['results']   
    pred_dt = results["pred_dt"]
    pred_tf = results["pred_tf"]

    return render_template('prediction.html', pred_dt=pred_dt, pred_tf=pred_tf)
 

@app.route("/get_brands/<brand>", methods=["GET"])
def get_brands(brand):
    models = brands_models[brand]
    # do something with the selected option
    return jsonify(models=models)  



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, port=port,
             #host='0.0.0.0'
             )