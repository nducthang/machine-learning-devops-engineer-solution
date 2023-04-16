'''
The library of functions to find customers who are likely to churn

Author: Thang Nguyen-Duc (ThangND34)
Date: Feb 27, 2023
'''


# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.rcParams["figure.figsize"] = 20, 10

IMAGES_EDA_FOLDER = "images/eda/"

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    _ = data_frame['Churn'].hist()
    plt.savefig(os.path.join(IMAGES_EDA_FOLDER, "churn_hist.png"))
    plt.clf()

    _ = data_frame['Customer_Age'].hist()
    plt.savefig(os.path.join(IMAGES_EDA_FOLDER, "customer_age_hist.png"))
    plt.clf()

    _ = data_frame.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    plt.savefig(os.path.join(IMAGES_EDA_FOLDER, "marital_status_hist.png"))
    plt.clf()

    _ = sns.histplot(
        data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(IMAGES_EDA_FOLDER, "total_trans_ct_hist.png"))
    plt.clf()

    _ = sns.heatmap(data_frame.corr(), annot=False,
                    cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(IMAGES_EDA_FOLDER, "corr_heatmap.png"))
    plt.clf()


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]
    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category_name in category_lst:
        category_lst = []
        category_groups = data_frame.groupby(category_name).mean()[response]
        for val in data_frame[category_name]:
            category_lst.append(category_groups.loc[val])
        data_frame[category_name + "_" + response] = category_lst
    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]

    x_data = pd.DataFrame()
    y_data = data_frame[response]
    x_data[keep_cols] = data_frame[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Classification reports
    rf_test_report = classification_report(y_test, y_test_preds_rf)
    rf_train_report = classification_report(y_train, y_train_preds_rf)

    lr_test_report = classification_report(y_test, y_test_preds_lr)
    lr_train_report = classification_report(y_train, y_train_preds_lr)

    # Create subplots
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Plot reports on subplots
    axs[0,
        0].text(0.01,
                0.99,
                "Training Set Logistic Regression\n\n" + lr_train_report,
                transform=axs[0,
                              0].transAxes,
                fontsize=10,
                verticalalignment='top')
    axs[0, 0].axis('off')
    axs[0,
        1].text(0.01,
                0.99,
                "Training Set Random Forest\n\n" + rf_train_report,
                transform=axs[0,
                              1].transAxes,
                fontsize=10,
                verticalalignment='top')
    axs[0, 1].axis('off')
    axs[1,
        0].text(0.01,
                0.99,
                "Testing Set Logistic Regression\n\n" + lr_test_report,
                transform=axs[1,
                              0].transAxes,
                fontsize=10,
                verticalalignment='top')
    axs[1, 0].axis('off')
    axs[1, 1].text(0.01, 0.99, "Testing Set Random Forest\n\n" + rf_test_report,
                   transform=axs[1, 1].transAxes, fontsize=10, verticalalignment='top')
    axs[1, 1].axis('off')

    # Save report
    plt.savefig("images/results/classification_report.png")
    plt.clf()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save images
    plt.savefig(os.path.join(output_pth, 'feature_importances.png'))
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Training models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Save classification report images
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Plot ROC and save to images/results folder
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_,
                       x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig("images/results/roc_curve.png")
    plt.clf()

    # Save feature importance plot
    feature_importance_plot(cv_rfc, x_test, "images/results")

    # Save models
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


if __name__ == '__main__':
    logger.info("Import data")
    DF = import_data("./data/bank_data.csv")
    logger.info("Perform EDA")
    perform_eda(DF)
    logger.info("Encode categorical columns")
    CATEGORY_LST = ["Gender",
                    "Education_Level",
                    "Marital_Status",
                    "Income_Category",
                    "Card_Category"]
    RESPONSE = "Churn"
    DF = encoder_helper(DF, CATEGORY_LST, RESPONSE)
    logger.info("Perform feature engineering")
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF, RESPONSE)
    logger.info("Train models ang get results")
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    logger.info("Done training")
