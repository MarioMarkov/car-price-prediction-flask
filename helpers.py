import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from requests import Request

def encode(data,ordinal_encoder, oh_encoder, oh_columns, ordinal_columns):
    data[ordinal_columns] = ordinal_encoder.transform(data[ordinal_columns])

    # One-Hot Encoding
    oh_columns_data = pd.DataFrame(oh_encoder.transform(data[oh_columns])) 

    # One-hot encoding removed index; put it back
    oh_columns_data.index = data.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_car = data.drop(oh_columns, axis=1)

    # Add one-hot encoded columns to numerical features
    data = pd.concat([num_X_car, oh_columns_data], axis=1)
    
    return data


def make_buckets(data):
  bins = [0,50000, 100000, 150000,200000,250000,300000,400000,500000,600000]
  labels = [1,2,3,4,5,6,7,8,9]
  data['kms'] = pd.cut(x = data['kms'], 
                     bins = bins,
                     labels = labels,
                     include_lowest = True)
  data["kms"] = data["kms"].astype("int32")
  return data

def basic_preprocessing(car: pd.DataFrame) -> pd.DataFrame:
  #car = car.drop(['euro'],axis='columns')
  premium_brands = ["Porsche", "Audi","Mercedes-Benz","BMW"]
  car['premium'] = np.where(car['brand'].isin(premium_brands), 1, 0)
  car = make_buckets(car)
  return car
   
def dt_model_preprocessing(car :pd.DataFrame) -> pd.DataFrame:
  car["brand"] = car['brand'].astype(str) +"-"+ car["model"]
  car = car.drop(['model'],axis='columns')


  car['new'] = np.where(car['year']>2018, 1, 0)

  # Ordinal Encoding
  ordinal_enc_cols = ['brand','color']
  one_hot_columns = ['fuel','type']  
    
  ordinal_encoder = pickle.load(open('pickles/ordinal_encoder', 'rb'))
  oh_encoder = pickle.load(open('pickles/onehot_encoder', 'rb'))
    
  car = encode(car, 
                    oh_encoder=oh_encoder,
                    ordinal_encoder=ordinal_encoder,
                    ordinal_columns=ordinal_enc_cols,
                    oh_columns=one_hot_columns)
  return car

def dt_model_prediction(car: pd.DataFrame) -> int:
  pickled_model = pickle.load(open('pickles/final_model_pickle.pkl', 'rb'))
  prediction = pickled_model.predict(car)  # features Must be in the form [[a, b]]
  prediction = int(prediction[0])

  return prediction


def tf_model_prediction(car: pd.DataFrame) -> int:
  tf_model = tf.keras.models.load_model('./saved_model/my_model')

  input_dict = {name: tf.convert_to_tensor([value]) for name, value in car.items()}
  prediction = tf_model.predict(input_dict)
  prediction = int(prediction[0])

  return prediction

def extract_features(request : Request)-> pd.DataFrame:

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
  #euro = float(request.form['euro'])

  car = pd.DataFrame({"brand": [brand],"model":[model] , "year": [year],"fuel": [fuel], "kms":[kms], 
    'transmission':  [transmission],'2door': [door_2],'color':[color],'type':[type_car],'displacement':[displacement],'hp':[hp],
    #'euro':[euro]
  })
  return car

def processing_for_categorical(car):
  model = XGBRegressor(enable_categorical=True,
                           tree_method="hist",
                     eval_metric="mae" ,
                     max_depth=5,
                     n_estimators =250,
                     colsample_bytree = 0.5,
                    max_cat_to_onehot=21,
  )
  model.load_model("xgb_boost_categorical.json")
  ## processing 
  car = make_buckets(car)
  car["kms"] = car["kms"].astype('int32')
  categorical_cols = list(car.select_dtypes(include='object'))
  car[categorical_cols] = car[categorical_cols].astype('category')
  car[["transmission","2door"]] = car[["transmission","2door"]].astype('int64')
  print(car)
  prediction = model.predict(car) 
  return prediction
def make_buckets_v2(data):
  bins = [0,50000, 100000, 150000,200000,250000,300000,400000,500000,600000,791100]
  labels = [1,2,3,4,5,6,7,8,9,10]
  data['kms'] = pd.cut(x = data['kms'],
                     bins = bins,
                     labels = labels,
                     include_lowest = True)
  return data
