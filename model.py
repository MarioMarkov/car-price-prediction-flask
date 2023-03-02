import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor


#import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


url='https://drive.google.com/file/d/1bCojj5ceFfy3u8mPS1BD4l7I3Id6Y2n5/view?usp=share_link'
data_url='https://drive.google.com/uc?id=' + url.split('/')[-2]

url = 'https://drive.google.com/file/d/1gTMe8OlV-uCdOS7tix7lFcDzlws-mv3i/view?usp=share_link'
test_url ='https://drive.google.com/uc?id=' + url.split('/')[-2]
# Load data
data = pd.read_csv(data_url, index_col=None)
test = pd.read_csv(test_url,  index_col=None)

# Remove Id column
data = data.drop(['id'],axis='columns')

# Dropping some columns
data = data.drop(['euro'],axis='columns')
test = test.drop(['euro'],axis='columns')

#@title Clean data 🧹 

print("Old Shape before cleaning: ", data.shape)

def clean_data(data):

  # Remove brands that are seen less than 200 times
  data = data.groupby('brand').filter(lambda x :len(x)>200)

  #Format BMW model
  def format_bmw_model(model_name):
    if 'X' in model_name or 'i' in model_name:
      return model_name
    return model_name[0]

  # Trim model to just 1 letter except if it is X or i ex.(318 to 3)
  data.loc[data['brand'] == 'BMW', ['model']] = data[data.brand == 'BMW'].model.apply(lambda x: format_bmw_model(x))

  # Remove models that are met less than 9 times
  data = data.groupby('model').filter(lambda x :len(x)>20)

  # Remove years that are met less than 10 times
  data = data.groupby('year').filter(lambda x :len(x)>10)

  #data.dropna(how="any", axis=0, inplace=True)

  # TODO this removes missing values 
  data = data[(data.hp >30) & (data.hp < 480)]
  data = data[(data.price >100) & (data.price < 60000)]
  data = data[(data.displacement >100) & (data.displacement < 8000)]
  data = data[(data.kms >100) & (data.kms < 1000000)]
  data = data.dropna()

  return data 

# Applying cleaning
data = clean_data(data)

print("New Shape after cleaning: ", data.shape)

# Put a flag on premium brands
# Get brands and models dictionary
brands_models = {}
brands = data['brand'].unique().tolist()
for brand in brands:
    # Get a list of models for the current brand
    models = pd.unique(data[data['brand'] == brand]['model'].tolist())
    # Add the brand and its models to the dictionary
    brands_models[brand] = models.tolist()

with open("brands_models.json", "w") as file:
    json.dump(brands_models, file)


ordinal_enc_cols = ['brand','model','color']
one_hot_columns = ['fuel','type']

# # Fit ordinal encoder
ordinal_encoder = OrdinalEncoder()
data[ordinal_enc_cols] = ordinal_encoder.fit_transform(data[ordinal_enc_cols])
test[ordinal_enc_cols] = ordinal_encoder.transform(test[ordinal_enc_cols])

# Apply one-hot encoder to fuel column
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_columns_data = pd.DataFrame(OH_encoder.fit_transform(data[one_hot_columns]))
oh_columns_test = pd.DataFrame(OH_encoder.transform(test[one_hot_columns])) 

# One-hot encoding removed index; put it back
oh_columns_data.index = data.index
oh_columns_test.index = test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_data = data.drop(one_hot_columns, axis=1)
num_X_test = test.drop(one_hot_columns, axis=1)

# Add one-hot encoded columns to numerical features
data = pd.concat([num_X_data, oh_columns_data], axis=1)
test = pd.concat([num_X_test, oh_columns_test], axis=1)

# Train set without price variableß
X = data.drop(['price'],axis='columns')

# Train set price variable
y = data.price

X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.25, random_state=42)



model = XGBRegressor(random_state=1,
                     eval_metric="mae",
                     objective='reg:squarederror',
                     learning_rate =0.1,
                         max_depth = 6,
                         colsample_bytree = 0.5,
                         reg_alpha = 1,
                         reg_lambda = 0.1,
                         n_estimators =250,
                     gamma = 1,
                     min_child_weight = 1)


model.fit(X_train, y_train,eval_set=[(X_test, y_test)], verbose=False)


print("Model score: ",model.score(X_train,y_train))

# Calculate error 
mae_train = -1 * cross_val_score(model, X_train, y_train,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')


mae_test = -1 * cross_val_score(model, X_test, y_test,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')

# Supress scientific notation
pd.options.display.float_format = '{:.10f}'.format

print("Mean absolute error CV train: ", mae_train.mean())

print("Mean absolute error CV test: ", mae_test.mean())

# LGBM model
import lightgbm as lgbm
import catboost

lgbm_model = lgbm.LGBMRegressor(n_estimators=250 ,
                                colsample_bytree = 1,   
                                learning_rate =0.1,
                                max_depth= 8 )

lgbm_model.fit(X_train, y_train, 
               eval_set=[(X_test, y_test)],
               eval_metric="rmse",
               early_stopping_rounds=50,
               verbose=-1)



pickle.dump(model, open('model.pkl','wb'))

with open("onehot_encoder", "wb") as f: 
    pickle.dump(OH_encoder, f)

with open("ordinal_encoder", "wb") as f: 
    pickle.dump(ordinal_encoder, f)

