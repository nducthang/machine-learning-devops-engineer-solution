from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data, outdated_packages_list
from scoring import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
# app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def prediction():        
    #call the prediction function you created in Step 3
    dataset_path = request.form.get('path')
    preds = model_predictions(deployed_model_path, dataset_path[1:-1])
    return  json.dumps([int(item) for item in preds]) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1 = score_model(deployed_model_path, test_data_path)
    return json.dumps(f1)#add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    df_stats = dataframe_summary(dataset_csv_path)
    return json.dumps(df_stats.to_dict())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    time_run = execution_time()
    na_percents = missing_data(dataset_csv_path)
    # if os.path.isfile('dependencies.json'):
    #     dependencies = json.load(open('dependencies.json'))
    # else:
    dependencies = outdated_packages_list().to_dict('records')
    diagnose_dict = {
        'time_run': time_run,
        'na_percents': na_percents,
        'dependencies': dependencies
    }
    return json.dumps(diagnose_dict)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
