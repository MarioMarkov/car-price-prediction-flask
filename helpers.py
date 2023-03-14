import pandas as pd

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