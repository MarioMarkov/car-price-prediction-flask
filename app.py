import numpy as np
from flask import Flask, request, render_template
import pickle
import sys
import pandas as pd
import json
import math
from flask import jsonify



#Create an app object using the Flask class. 
app = Flask(__name__)
app.static_folder = 'static'
#Load the trained model. (Pickle file)
with open("brands_models.json", "r") as file:
    brands_models = json.load(file)

brands  = list(brands_models.keys())

@app.route('/')
def home():
    return render_template('index.html', len = len(brands), brands = brands)


@app.route('/predict',methods=['POST'])
def predict():
    pickled_model = pickle.load(open('model.pkl', 'rb'))

    brand = str(request.form['brand'])
    model = str(request.form['model'])
    year = int(request.form['year'])
    fuel = str(request.form['fuel'])
    kms = float(request.form['kms'])
    transmission = float(request.form['transmission'])
    door_2 = float(request.form['2door'])
    color = str(request.form['color'])
    type_car = str(request.form['type'])
    displacement = float(request.form['displacement'])
    hp = float(request.form['hp'])
    euro = float(request.form['euro'])

    car = pd.DataFrame({"brand": [brand],"model":[model] , "year": [year],"fuel": [fuel], "kms":[kms], 
    'transmission':  [transmission],'2door': [door_2],'color':[color],'type':[type_car],'displacement':[displacement],'hp':[hp],'euro':[euro]
    })

    # Ordinal Encoding

    ordinal_enc_cols = ['brand','model','color','type']
    ordinal_encoder = pickle.load(open('ordinal_encoder', 'rb'))
    car[ordinal_enc_cols] = ordinal_encoder.transform(car[ordinal_enc_cols])


    # One-Hot Encoding
    one_hot_columns = ['fuel']
    oh_encoder = pickle.load(open('onehot_encoder', 'rb'))
    oh_columns = pd.DataFrame(oh_encoder.transform(car[one_hot_columns])) 

    # One-hot encoding removed index; put it back
    oh_columns.index = car.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_car = car.drop(one_hot_columns, axis=1)

    # Add one-hot encoded columns to numerical features
    car = pd.concat([num_X_car, oh_columns], axis=1)
    prediction = pickled_model.predict(car)  # features Must be in the form [[a, b]]

    output = np.exp(prediction[0])
    output = int(output)
    return render_template('index.html', len = len(brands), brands = brands, prediction_text='Price  is {}'.format(output))

@app.route("/get_brands/<brand>", methods=["GET"])
def get_brands(brand):
    models = brands_models[brand]
    # do something with the selected option
    return jsonify(models=models)  

if __name__ == "__main__":
    app.run(debug=True)