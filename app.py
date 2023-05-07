import json
import pickle
import helpers
import numpy as np
import pandas as pd
import tensorflow as tf
from vertex_model import get_value_vertex_model
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
    
    #pickled_model = pickle.load(open('pickles/xgb_model.pkl', 'rb'))
    # With joblib it does not work 
    # Important to install scikit-learn v 1.2.1
    tf_model = tf.keras.models.load_model('./saved_model/my_model')

    pickled_model = pickle.load(open('pickles/final_model_pickle.pkl', 'rb'))


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
    #ml_model = str(request.form['ml_model_hidden'])

    car = pd.DataFrame({"brand": [brand],"model":[model] , "year": [year],"fuel": [fuel], "kms":[kms], 
    'transmission':  [transmission],'2door': [door_2],'color':[color],'type':[type_car],'displacement':[displacement],'hp':[hp],'euro':[euro]
    })
    
    car_vertex = {
        "instances": [
            { 
            "_2door": door_2,"brand": brand,"color": color,"displacement": int(displacement),
            "fuel": fuel,"hp": int(hp),"kms": kms,"model": model, "transmission": transmission,
            "type": type_car,"year": str(year)
            }
        ]
    }
    print(car_vertex)
    
    car = car.drop(['euro'],axis='columns')
    car["brand"] = car['brand'].astype(str) +"-"+ car["model"]

    car = car.drop(['model'],axis='columns')

    premium_brands = ["Porsche", "Audi","Mercedes-Benz","BMW"]

    car['premium'] = np.where(car['brand'].isin(premium_brands), 1, 0)
    
    car = helpers.make_buckets(car)

    car['new'] = np.where(car['year']>2018, 1, 0)

    car_tf = car.copy()

    # Flag for new cars only for decission trees
    
    # Ordinal Encoding
    ordinal_enc_cols = ['brand','color']
    one_hot_columns = ['fuel','type']  
    
    ordinal_encoder = pickle.load(open('pickles/ordinal_encoder', 'rb'))
    oh_encoder = pickle.load(open('pickles/onehot_encoder', 'rb'))
    
    car = helpers.encode(car, 
                    oh_encoder=oh_encoder,
                    ordinal_encoder=ordinal_encoder,
                    ordinal_columns=ordinal_enc_cols,
                    oh_columns=one_hot_columns)
    
    car_tf = helpers.encode(car_tf, 
                    oh_encoder=oh_encoder,
                    ordinal_encoder=ordinal_encoder,
                    ordinal_columns=ordinal_enc_cols,
                    oh_columns=one_hot_columns)
    

    prediction = pickled_model.predict(car)  # features Must be in the form [[a, b]]
    prediction_tf = tf_model.predict(car_tf)


    

    output = int(prediction[0])
    output_tf = int(prediction_tf[0])
    output_vertex = int(get_value_vertex_model(car_vertex))

    return render_template(
        'index.html', 
        len = len(brands),
        brands = brands, 
        prediction_text='Price  is '+ str(output)+ 'and tf_model' +str(output_tf)+ ' Vertex: '+
        str(output_vertex))

@app.route("/get_brands/<brand>", methods=["GET"])
def get_brands(brand):
    models = brands_models[brand]
    # do something with the selected option
    return jsonify(models=models)  

if __name__ == "__main__":
    app.run(debug=True)