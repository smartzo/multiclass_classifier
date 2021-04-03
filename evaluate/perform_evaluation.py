import numpy as np
import pandas as pd
import xgboost
import torch

from config import APPLY_STANDARD_SCALER
from config import FEATURES, TARGET
from config import TARGET_COLUMN_NAME, TIME_COLUMN_NAME
from config import XGBOOST, PYTORCH

########################################################################################################################


class EvaluationDataClass:
    def __init__(self, X, y_true, standard_scaler):
        self.TrueScale = {FEATURES: X, TARGET: y_true}
        self.standard_scaler = standard_scaler

    def apply_standard_scaling(self):

        X = self.TrueScale[FEATURES]
        X_scaled = np.zeros_like(X)
        standard_scaler = self.standard_scaler[FEATURES]

        n_columns = X.shape[1]
        for j in range(n_columns):
            mu = standard_scaler.mean[j]
            sigma = standard_scaler.std[j]

            X_scaled[:, j] = (X[:, j] - mu) / sigma

        return X_scaled

    def invert_standard_scaling(self, y_pred):
        mu = self.standard_scaler[TARGET].mean
        sigma = self.standard_scaler[TARGET].std

        y_pred = sigma * y_pred + mu

        return y_pred

########################################################################################################################


def _separate_features_and_target(data_frame, standard_scaler):

    # --------------------------------------------------------------------------------------------------------------

    features = data_frame.drop(TARGET_COLUMN_NAME, axis=1)

    features = features.drop(TIME_COLUMN_NAME, axis=1)

    target = data_frame[TARGET_COLUMN_NAME]

    # --------------------------------------------------------------------------------------------------------------

    X = np.array(features, dtype=float)

    y = np.array(target, dtype=float)

    # --------------------------------------------------------------------------------------------------------------

    eval_data = EvaluationDataClass(X, y, standard_scaler)

    return eval_data

########################################################################################################################


def _get_predictions(model_name, model, eval_data):

    def predict_xgboost(X):
        D = xgboost.DMatrix(X, label=y_true)
        y_pred = model.predict(D)
        y_pred = np.rint(y_pred)
        return y_pred

    def predict_pytorch(X):
        X = (torch.from_numpy(X).float()).to(torch.device("cpu"))
        y_pred = model(X)
        y_pred = y_pred.argmax(dim=1, keepdim=True)
        y_pred = y_pred.detach().numpy()
        y_pred = y_pred.reshape((y_pred.shape[0],))
        return y_pred

    y_pred = None

    if APPLY_STANDARD_SCALER[model_name]:
        X = eval_data.apply_standard_scaling()
    else:
        X = eval_data.TrueScale[FEATURES]

    y_true = eval_data.TrueScale[TARGET]

    if XGBOOST in model_name:
        y_pred = predict_xgboost(X)
    elif PYTORCH in model_name:
        y_pred = predict_pytorch(X)

    # if APPLY_STANDARD_SCALER[model_name]:
    #     y_pred = eval_data.invert_standard_scaling(y_pred)

    return y_pred

########################################################################################################################


def execute(fold, eval_manager, model_name, model, eval_dataset_df, eval_dataset_type, standard_scaler, global_output):

    # ------------------------------------------------------------------------------------------------------------------

    eval_data = _separate_features_and_target(eval_dataset_df, standard_scaler)

    y_pred = _get_predictions(model_name, model, eval_data)

    eval_manager.dump_predictions(eval_dataset_type, model_name, fold, y_pred)

    y_true = eval_data.TrueScale[TARGET]

    eval_manager.scores.assign(y_true, y_pred)

    # ------------------------------------------------------------------------------------------------------------------

    bacc = round(eval_manager.scores.BACC.regr, 2)
    accr = round(eval_manager.scores.ACCR.regr, 2)
    output = pd.DataFrame([[fold, model_name, bacc, accr]], columns=global_output.columns)
    global_output = global_output.append(output, ignore_index=True)

    # ------------------------------------------------------------------------------------------------------------------

    return global_output

########################################################################################################################
