import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    df_input = []
    lst_files  = []
    for file_path in glob.glob(os.path.join(input_folder_path, '*.csv')):
        df = pd.read_csv(file_path)
        df_input.append(df)
        lst_files.append(file_path+"\n")

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    df_result = pd.concat(df_input, axis=0, ignore_index=False).drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.writelines(lst_files)



if __name__ == '__main__':
    merge_multiple_dataframe()
