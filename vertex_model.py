import os
import subprocess
import requests
import json
import pandas as pd
#os.system("gcloud auth application-default login")

def get_value_vertex_model(car_object,file_name=''):
    ENDPOINT_ID="3864559071260573696"
    PROJECT_ID="934044731923"
    #INPUT_DATA_FILE=file_name
   
    command1 = subprocess.run('gcloud auth print-access-token', shell=True, capture_output=True, text=True).stdout.replace('\n', '')
    headers = {
        'Authorization': 'Bearer ' + command1,
        'Content-Type': 'application/json',
    }

    # with open(INPUT_DATA_FILE) as f:
    #     data = f.read().replace('\n', '').replace('\r', '').encode()


    response = requests.post(
        'https://us-central1-aiplatform.googleapis.com/v1/projects/' + PROJECT_ID + '/locations/us-central1/endpoints/' +ENDPOINT_ID + ':predict',
        headers=headers,
        json=car_object
    )

    # Use the json module to load CKAN's response into a dictionary.
    response_values = json.loads(response.text)["predictions"]
    value = response_values[0]["value"]
    return value


# car = pd.DataFrame({"brand": ["Opel"],"model":["Corsa"] , "year": [year],"fuel": [fuel], "kms":[kms], 
#     'transmission':  [transmission],'2door': [door_2],'color':[color],'type':[type_car],'displacement':[displacement],'hp':[hp],'euro':[euro]
#     })

car = {
        "instances": [
        { "_2door": 0.0,"brand": "VW","color": "gray","displacement": 2000,"fuel": "d","hp": 120,"kms": 195000.0,"model": "Golf","transmission": 0.0,"type": "hatchback","year": "2020"}
        ]
}
print(get_value_vertex_model(car))

