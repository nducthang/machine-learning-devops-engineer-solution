
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
dataset_csv_path = os.path.join(config['output_folder_path'], "finaldata.csv") 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 

##################Function to get model predictions
def model_predictions(deployed_model_path, test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    model = pickle.load(open(deployed_model_path, 'rb'))
    df_test = pd.read_csv(test_data_path)
    X_test = df_test.drop(['corporation', 'exited'], axis='columns')
    preds = model.predict(X_test)
    return preds

##################Function to get summary statistics
def dataframe_summary(data_path):
    #calculate summary statistics here
    df = pd.read_csv(data_path)
    df_stats = df.describe().iloc[1:3]
    median_list = []
    for col in df_stats.columns:
        median_list.append(df[col].median(axis=0))
    df_median = pd.DataFrame([median_list], columns=df_stats.columns, index=['median'])
    df_stats = pd.concat([df_stats, df_median])
    return df_stats


##################Function to Missing Data
def missing_data(data_path):
    df = pd.read_csv(data_path)
    na_list = list(df.isna().sum(axis=0))
    na_percents = [na_list[i]/len(df.index) for i in range(len(na_list))]
    return na_percents

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time
    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of current and latest versions of modules used in this script
    if not os.path.isfile('requirements.txt'):
        subprocess.check_output(['pip', 'freeze'])

    with open('requirements.txt', 'r') as f:
        modules = f.read().splitlines()
    
    modules_dict = {'module_name': [], 'current_version': [], 'latest_version': []}
    for module in modules:
        module_name, current_version = module.split('==')
        latest_version = subprocess.check_output(['pip', 'index', 'versions', module_name])
        latest_version = latest_version.split(b'versions: ')[1].split(b', ')[0]
        latest_version = latest_version.decode('ascii')
        modules_dict['module_name'].append(module_name)
        modules_dict['current_version'].append(current_version)
        modules_dict['latest_version'].append(latest_version)
        
    df_results = pd.DataFrame(modules_dict)

    return df_results


if __name__ == '__main__':
    print("Model predictions", model_predictions(deployed_model_path, test_data_path))
    print("Dataframe Summary:\n", dataframe_summary(dataset_csv_path))
    print("Missing data:", missing_data(dataset_csv_path))
    print("Execution time:", execution_time())
    print("check dependencies:\n", outdated_packages_list())
