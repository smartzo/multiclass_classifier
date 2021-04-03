import numpy as np
import xgboost
import torch
# from statistics import mode
from scipy import stats

from config import FEATURES, TARGET, TARGET_COLUMN_NAME
from config import XGBOOST, PYTORCH
from config import APPLY_STANDARD_SCALER

########################################################################################################################


class DataClass:

    def __init__(self, X, y_true, standard_scaler):
        self.TrueScale = {FEATURES: X, TARGET: y_true}
        self.standard_scaler = standard_scaler
        X_size = np.shape(X)
        self.n_datapoints = X_size[0]
        self.n_features = X_size[1]

    def apply_standard_scaling(self, model_num):

        X = self.TrueScale[FEATURES]
        X_scaled = np.zeros_like(X)
        standard_scaler = self.standard_scaler[model_num][FEATURES]

        for j in range(self.n_features):
            mu = standard_scaler.mean[j]
            sigma = standard_scaler.std[j]

            X_scaled[:, j] = (X[:, j] - mu) / sigma

        return X_scaled

    def invert_standard_scaling(self, y_pred, model_num):
        mu = self.standard_scaler[model_num][TARGET].mean
        sigma = self.standard_scaler[model_num][TARGET].std

        y_pred = sigma * y_pred + mu

        return y_pred

########################################################################################################################


class AggregatedPredictionsClass:

    def __init__(self, n_datapoints):
        self.n_datapoints = n_datapoints
        self.per_algorithm = {}
        self.combined_algorithms = []

    def add_algorithm(self, model_name):
        self.per_algorithm[model_name] = []
        return

    def aggregate_per_algorithm(self, model_name, y_pred):
        self.per_algorithm[model_name].append(y_pred)
        return

    def aggregate_combined_algorithms(self, algorithm_name):
        y_pred = self.per_algorithm[algorithm_name]
        self.combined_algorithms.append(y_pred)
        return

    def majority_voting(self, model_name):

        def generic_vote(stacked_lists):

            vote = np.zeros(n_datapoints)

            n_stacks = len(stacked_lists)

            local_vote = np.zeros(n_stacks)

            for i in range(n_datapoints):
                for k in range(n_stacks):
                    local_vote[k] = stacked_lists[k][i]
                # vote[i] = mode(local_vote)
                vote[i] = int(stats.mode(local_vote)[0])

            return vote

        n_datapoints = self.n_datapoints

        if model_name is not None:
            stacked_predictions = self.per_algorithm[model_name]
            self.per_algorithm[model_name] = generic_vote(stacked_predictions)
        else:
            stacked_predictions = self.combined_algorithms
            self.combined_algorithms = generic_vote(stacked_predictions)

        return

########################################################################################################################


def _apply_data_scaler(model_name, model_num, data_info):

    if APPLY_STANDARD_SCALER[model_name]:
        X = data_info.apply_standard_scaling(model_num)
    else:
        X = data_info.TrueScale[FEATURES]

    return X

########################################################################################################################


def _compute_predictions(model, model_name, X):

    y_pred = None

    if model_name == XGBOOST:
        D = xgboost.DMatrix(X, label=None)
        y_pred = model.predict(D)
    elif model_name == PYTORCH:
        X = (torch.from_numpy(X).float()).to(torch.device("cpu"))
        y_pred = model(X)
        y_pred = y_pred.argmax(dim=1, keepdim=True)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.reshape((y_pred.shape[0],))
        return y_pred

    return y_pred

########################################################################################################################


def _predict(model_info, data_info):

    # ------------------------------------------------------------------------------------------------------------------

    n_datapoints = data_info.n_datapoints

    aggregated_predictions = AggregatedPredictionsClass(n_datapoints)

    my_model = model_info.model

    # ------------------------------------------------------------------------------------------------------------------

    for model_name in my_model:

        aggregated_predictions.add_algorithm(model_name)

        for model_num in my_model[model_name]:

            X = _apply_data_scaler(model_name, model_num, data_info)

            model = my_model[model_name][model_num]

            y_pred = _compute_predictions(model, model_name, X)

            aggregated_predictions.aggregate_per_algorithm(model_name, y_pred)

        aggregated_predictions.majority_voting(model_name)

        aggregated_predictions.aggregate_combined_algorithms(model_name)

    aggregated_predictions.majority_voting(None)

    # ------------------------------------------------------------------------------------------------------------------

    y_pred = aggregated_predictions.combined_algorithms

    return y_pred

########################################################################################################################


def execute(holdout_info, model_info):

    holdout_matrix = holdout_info.matrix

    X = holdout_matrix.to_numpy()

    data_info = DataClass(X, None, model_info.standard_scaler)

    predictions = _predict(model_info, data_info)

    predictions = [holdout_info.label2status[x] for x in predictions]

    holdout_info.timestamp2prediction[TARGET_COLUMN_NAME] = predictions

    return predictions

########################################################################################################################
