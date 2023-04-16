'''
The unit tests for the churn_library.py function

Author: Thang Nguyen-Duc (ThangND34)
Date: Feb 27, 2023
'''

import os
import logging
import glob
import sys
import joblib
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="module")
def file_path():
    '''
    file_path fixture - returns file_path to the dataftame
    '''
    return "./data/bank_data.csv"


@pytest.fixture(scope="module")
def data_frame_raw(file_path):
    '''
    dataframe raw fixture - returns dataframe raw from origin file
    '''
    try:
        data_frame = cls.import_data(file_path)
        logging.info("Create dataframe raw: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing data_frame_raw: The file wasn't found")
        raise err
    return data_frame


@pytest.fixture(scope="module")
def data_frame_encoder(data_frame_raw):
    '''
    dataframe encoder - returns dataframe after add new columns
    '''
    try:
        category_lst = ["Gender",
                        "Education_Level",
                        "Marital_Status",
                        "Income_Category",
                        "Card_Category"]
        data_frame_encoder = cls.encoder_helper(data_frame_raw,
                                                category_lst=category_lst,
                                                response="Churn")
        logging.info("Create dataframe encoder: SUCCESS")
    except KeyError as err:
        logging.error("Testing data_frame_encoder: Not exits columns")
        raise err
    return data_frame_encoder


@pytest.fixture(scope="module")
def feature_engineering(data_frame_encoder):
    '''
    create data after feature engineering - returns: x_train, x_test, y_train, y_test
    '''
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            data_frame_encoder,
            response="Churn"
        )
        logging.info("Create dataframe feature engineering: SUCCESS")
    except BaseException as err:
        logging.error("Testing create dataframe feature engineering error")
        raise err

    return x_train, x_test, y_train, y_test


def test_import(data_frame_raw):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        assert data_frame_raw.shape[0] > 0
        assert data_frame_raw.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(data_frame_raw):
    '''
    test perform eda function - The graphic must be exits in images folder
    '''
    cls.perform_eda(data_frame_raw)
    try:
        assert os.path.isfile(os.path.join(
            cls.IMAGES_EDA_FOLDER, "churn_hist.png"))
        assert os.path.isfile(os.path.join(
            cls.IMAGES_EDA_FOLDER, "customer_age_hist.png"))
        assert os.path.isfile(os.path.join(
            cls.IMAGES_EDA_FOLDER, "marital_status_hist.png"))
        assert os.path.isfile(os.path.join(
            cls.IMAGES_EDA_FOLDER, "total_trans_ct_hist.png"))
        assert os.path.isfile(os.path.join(
            cls.IMAGES_EDA_FOLDER, "corr_heatmap.png"))
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The files were not found")
        raise err


def test_encoder_helper(data_frame_encoder):
    '''
    test encoder helper - Check that new columns exits
    '''
    try:
        assert data_frame_encoder.shape[0] > 0
        assert data_frame_encoder.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder helper: The files have not columns and rows")
        raise err

    try:
        for column in ["Gender",
                       "Education_Level",
                       "Marital_Status",
                       "Income_Category",
                       "Card_Category"]:
            assert column in data_frame_encoder
    except AssertionError as err:
        logging.error("Testing encoder_helper: The new columns not exits")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return data_frame_encoder


def test_perform_feature_engineering(feature_engineering):
    '''
    test perform_feature_engineering - Checks len of features
    '''
    x_train = feature_engineering[0]
    x_test = feature_engineering[1]
    y_train = feature_engineering[2]
    y_test = feature_engineering[3]
    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Sequence length mismatch")
        raise err


def test_train_models(feature_engineering):
    '''
    test train_models -check saved results after train models
    '''
    cls.train_models(feature_engineering[0], feature_engineering[1],
                     feature_engineering[2], feature_engineering[3])
    try:
        joblib.load('./models/rfc_model.pkl')
        joblib.load('./models/logistic_model.pkl')
        logging.info(
            "Test train_models - File checkpoints: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Test train_models - File checkpoints of models NOT exits")
        raise err

    for image_name in ["classification_report",
                       "feature_importances",
                       "roc_curve"
                       ]:
        try:
            assert os.path.isfile(f"images/results/{image_name}.png")
        except AssertionError as err:
            logging.error("Test train_models: File images were not found")
            raise err
    logging.info("Testing train_models - File results: SUCCESS")


if __name__ == "__main__":
    for directory in ["logs", "images/eda", "images/results", "models"]:
        files = glob.glob(f"{directory}/*")
        for file in files:
            os.remove(file)
    sys.exit(pytest.main(["-s"]))
