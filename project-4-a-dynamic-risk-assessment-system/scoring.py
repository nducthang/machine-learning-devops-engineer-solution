from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path =  config['output_model_path']
dataset_csv_path = os.path.join(config['output_folder_path'],) 
test_data_path = os.path.join(config['test_data_path'], "testdata.csv") 
model_path = os.path.join(output_model_path, 'trainedmodel.pkl')


#################Function for model scoring
def score_model(model_path, test_data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    print("Calculating F1 score...")
    print("Model path:", model_path)
    print("Test data path:", test_data_path)
    model = pickle.load(open(model_path, 'rb'))
    df = pd.read_csv(test_data_path)
    X_test = df.drop(['corporation', 'exited'], axis='columns')
    y_test = df['exited']
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    if not os.path.isdir(output_model_path):
        os.mkdir(output_model_path)
    with open(os.path.join(output_model_path, "latestscore.txt"), 'w') as f:
        f.write(str(f1))
    
    return f1

if __name__ == '__main__':
    print(score_model(model_path, test_data_path))