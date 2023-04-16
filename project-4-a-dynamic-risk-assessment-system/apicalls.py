import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://192.168.0.105:8000/"
with open('config.json','r') as f:
    config = json.load(f) 


test_data_file_path = os.path.join(config['test_data_path'], 'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.post(URL + 'prediction', data={'path': json.dumps(test_data_file_path)})
response2 = requests.get(URL + 'scoring')
response3 = requests.get(URL + 'summarystats')
response4 = requests.get(URL + 'diagnostics')

#combine all API responses
responses = {
    'prediction': response1.json(),
    'scoring': response2.json(),
    'summarystats': response3.json(),
    'diagnostics': response4.json()
}

#write the responses to your workspace
with open(os.path.join(config["output_model_path"], "apireturns2.txt"), 'w') as f:
    json.dump(responses, f)
