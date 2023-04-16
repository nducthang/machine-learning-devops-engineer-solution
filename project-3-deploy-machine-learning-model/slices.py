import os
import pandas as pd
import pickle
from ml.model import inference, compute_model_metrics
from ml.data import process_data


def compute_slice_metrics(model, encoder, lb, cleaned_df, categorical_features, slice_features):
    """
    Computes the model metrics for each slice of data that has a particular value for a given feature.
    Inputs
    ------
    model: ???
        model trained
    cleaned_df : pd.DataFrame
        Cleaned dataframe.
    categorical_features : list[str]
        List of the names of the categorical features.
    slice_features : str
         Name of the feature used to make slices (categorical features)

    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    None
    """

    slice_metrics = {}
    for value in cleaned_df[slice_features].unique():
        X_slice = cleaned_df[cleaned_df[slice_features] == value]
        X_slice, y_slice, _, _ = process_data(
            X_slice, categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
        preds = inference(model, X_slice)
        print(
            f"shape of preds: {preds.shape} & shape of y_slice: {y_slice.shape}")
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slice_metrics[value] = {'Precision': precision,
                                'Recall': recall, 'Fbeta': fbeta}
        print(
            f"Slice metrics for {slice_features} = {value}: {slice_metrics[value]}")

    # Write results to slice_output.txt
    with open('slice_output.txt', 'w') as f:
        for key, value in slice_metrics.items():
            f.write(f"{slice_features} = {key}: {value}")
            f.write("\n")
    return slice_metrics


if __name__ == '__main__':
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    file_dir = os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(file_dir, './data/census_cleaned.csv'))

    model_path = os.path.join(file_dir, './model/rf_model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    encoder_path = os.path.join(file_dir, './model/encoder.pkl')
    encoder = pickle.load(open(encoder_path, 'rb'))

    lb_path = os.path.join(file_dir, './model/lb.pkl')
    lb = pickle.load(open(lb_path, 'rb'))

    compute_slice_metrics(
        model=model,
        encoder=encoder,
        lb=lb,
        cleaned_df=data,
        categorical_features=cat_features,
        slice_features='education'
    )
