import pickle
import requests
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 


#Load data
data = pd.read_csv('./cars-data3.csv', index_col=None)
test = pd.read_csv('./test-data.csv',  index_col=None)


# Remove Id column
data = data.drop(['id'],axis='columns')


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
data = data.groupby('model').filter(lambda x :len(x)>9)


# Impute columns records with missing values with median or mode
data.kms.fillna(data.kms.median(), inplace = True)

# TODO maybe not right 
data = data.fillna(data.mode().iloc[0])

# Remove outliers in IQR 
Q3 = np.quantile(data.price, 0.95)
Q1 = np.quantile(data.price, 0.10)
IQR = Q3 - Q1
lower_range = Q1 - 1.5 * IQR
upper_range = Q3 + 1.5 * IQR
outlier_free_list = [x for x in data.price if (
    (x > lower_range) & (x < upper_range))]
data = data.loc[data.price.isin(outlier_free_list)]



# There are some cars with inpossibly low kms maybe add 100 thausand for each
Q3 = np.quantile(data.kms, 0.95)
Q1 = np.quantile(data.kms, 0.40)
IQR = Q3 - Q1
lower_range = Q1 - 1.5 * IQR
upper_range = Q3 + 1.5 * IQR
outlier_free_list = [x for x in data.kms if (
    (x > lower_range) & (x < upper_range))]
data = data.loc[data.kms.isin(outlier_free_list)]

data.price = np.log(data.price)

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


# Encoding string columns to numeric

ordinal_enc_cols = ['brand','model','color','type']
one_hot_columns = ['fuel','euro']

ordinal_encoder = OrdinalEncoder()
data[ordinal_enc_cols] = ordinal_encoder.fit_transform(data[ordinal_enc_cols])
test[ordinal_enc_cols] = ordinal_encoder.transform(test[ordinal_enc_cols])

with open("ordinal_encoder", "wb") as f: 
    pickle.dump(ordinal_encoder, f)

#scatter_matrix(data[['price','kms','year']], figsize=(12, 8))
#sns.catplot(data=data, x="fuel", y="price",kind="box")

# Apply one-hot encoder to fuel column
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
oh_columns_data = pd.DataFrame(OH_encoder.fit_transform(data[one_hot_columns]))
oh_columns_test = pd.DataFrame(OH_encoder.transform(test[one_hot_columns])) 

with open("onehot_encoder", "wb") as f: 
    pickle.dump(OH_encoder, f)

# One-hot encoding removed index; put it back
oh_columns_data.index = data.index
oh_columns_test.index = test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_data = data.drop(one_hot_columns, axis=1)
num_X_test = test.drop(one_hot_columns, axis=1)

# Add one-hot encoded columns to numerical features
data = pd.concat([num_X_data, oh_columns_data], axis=1)
test = pd.concat([num_X_test, oh_columns_test], axis=1)


# Train set without price col
X = data.drop(['price'],axis='columns')

# Train set price col
y = data.price


xgb_model = XGBRegressor(random_state=1,objective='reg:squarederror',
                         learning_rate = 0.07,
                         max_depth = 3,
                         colsample_bytree = 0.4,
                         n_estimators = 150)



# Fit model
xgb_model.fit(X,y)

#Calculate error 
mae = -1 * cross_val_score(xgb_model, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')


#Supress scientific notation
pd.options.display.float_format = '{:.10f}'.format


print(mae.mean())



submission_predictions =  xgb_model.predict(test)
submission_predictions = np.exp(submission_predictions)
print(submission_predictions)
pickle.dump(xgb_model, open('model.pkl','wb'))

