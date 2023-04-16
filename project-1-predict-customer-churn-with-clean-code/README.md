# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Author: Thang Nguyen-Duc

This is a project to implement best coding practices. 

We have already provided you the `churn_notebook.ipynb` file containing the solution to identify credit card customers that are most likely to churn, but without implementing the engineering and software best practices.

I refactor the given churn_notebook.ipynb file following the best coding practices to generate these files:

- churn_library.py
- churn_script_logging_and_tests.py
- README.md

## Files and data description
```
.
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Given: Contains the code to be refactored
├── churn_library.py     # Define the functions
├── churn_script_logging_and_tests.py # ToDo: Finish tests and logs
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models
```
You need attention these files:
- `churn_library.py`: is a library of functions to find customers who are likely to churn.
- `churn_script_logging_and_tests.py`: Contain unit tests for the churn_library.py functions.

## Running Files
Firtly, create enviroment:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda activate conda create -n .venv python=3.8 
conda activate .venv
```
Next, install requirements:
```
pip install -r requirements.txt
```
Run code:
```
python churn_library.py
```
