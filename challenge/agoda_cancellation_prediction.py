from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator

import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename)
    full_data = full_data.dropna(subset=["booking_datetime", "checkin_date", "checkout_date"])
    features = full_data[[
        "no_of_adults",
        "no_of_children",
        "no_of_extra_bed",
        "no_of_room",
        "is_first_booking",
        "is_user_logged_in",
        "request_nonesmoke",
        "request_latecheckin",
        "request_highfloor",
        "request_largebed",
        "request_twinbeds",
        "request_airport",
        "request_earlycheckin",
        "guest_is_not_the_customer",
        "hotel_star_rating"
    ]]

    for name in ["booking_datetime", "checkin_date", "checkout_date"]:
        months = pd.to_numeric(full_data[name].str.slice(5, 7))
        features = features.join(pd.get_dummies(months, prefix=f"{name}_month_no_"))

    reference_datetime = pd.to_datetime(full_data["booking_datetime"])
    features["checkin_date"] = (pd.to_datetime(full_data["checkin_date"]) - reference_datetime).dt.total_seconds()
    features["checkout_date"] = (pd.to_datetime(full_data["checkout_date"]) - reference_datetime).dt.total_seconds()

    labels = full_data["cancellation_datetime"].notna()

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
