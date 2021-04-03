import os
import numpy as np
import pandas as pd
import joblib
import pickle

from config import PATH_TO_BEST_MODEL
from config import PATH_TO_HOLDOUT_SET
from config import PATH_TO_DROPPED_FEATURES_FILENAME
from config import PATH_TO_LABEL2STATUS_PKL
from config import PATH_TO_STANDARD_SCALER
from config import MODEL_DELIMITER
from config import FEATURES
from config import TIME_COLUMN_NAME
from config import TARGET_COLUMN_NAME

########################################################################################################################


class HoldoutDataClass:
    def __init__(self, holdout_matrix, label2status, timestamp2prediction):
        self.matrix = holdout_matrix
        self.label2status = label2status
        self.timestamp2prediction = timestamp2prediction

########################################################################################################################


class ModelClass:

    def __init__(self, standard_scaler, model):
        self.standard_scaler = standard_scaler
        self.model = model

########################################################################################################################


class StatisticsClass:

    def __init__(self):
        self.mean = None
        self.std = None

########################################################################################################################


def _load_model():

    # ------------------------------------------------------------------------------------------------------------------

    dir_names = os.listdir(PATH_TO_BEST_MODEL)
    if len(dir_names) > 1:
        quit("Error: Too many items under " + PATH_TO_BEST_MODEL)
    model_combination_names = dir_names[0]

    model_combination_path = os.path.join(PATH_TO_BEST_MODEL, model_combination_names)

    model_names = model_combination_names.split(MODEL_DELIMITER)

    # ------------------------------------------------------------------------------------------------------------------

    model = {}

    for model_name in model_names:
        model[model_name] = {}

    # ------------------------------------------------------------------------------------------------------------------

    verbose_model_names = os.listdir(model_combination_path)
    if len(verbose_model_names) == 0:
        quit("Error: No models under " + model_combination_path)

    for verbose_model_name in verbose_model_names:
        (model_name, model_num, _) = verbose_model_name.split(MODEL_DELIMITER)
        verbose_model_path = os.path.join(model_combination_path, verbose_model_name)
        if model_name not in model:
            quit("Error: Incorrect model " + verbose_model_name + " under " + verbose_model_path)
        model[model_name][model_num] = joblib.load(verbose_model_path)

    # ------------------------------------------------------------------------------------------------------------------

    return model

########################################################################################################################


def _load_standard_scaler():

    if not os.path.isdir(PATH_TO_STANDARD_SCALER):
        return None

    standard_scaler = {}

    filenames = os.listdir(PATH_TO_STANDARD_SCALER)

    for filename in filenames:
        experiment_num, data_type = filename.split(".")

        if experiment_num not in standard_scaler:
            standard_scaler[experiment_num] = {}

        experiment_standard_scaler = standard_scaler[experiment_num]

        experiment_standard_scaler[data_type] = StatisticsClass()

        filename_path = os.path.join(PATH_TO_STANDARD_SCALER, filename)

        statistics = np.loadtxt(filename_path, delimiter=";")

        if data_type == FEATURES:
            experiment_standard_scaler[data_type].mean = statistics[0, :]
            experiment_standard_scaler[data_type].std = statistics[1, :]
        else:
            experiment_standard_scaler[data_type].mean = statistics[0]
            experiment_standard_scaler[data_type].std = statistics[1]

    return standard_scaler

########################################################################################################################


def _load_holdout_matrix():

    holdout_matrix = pd.read_csv(
                                  PATH_TO_HOLDOUT_SET,
                                  encoding='utf-8',
                                  skiprows=0,
                                  sep=',',
                                  skipinitialspace=True,
                                  index_col=None,
                                  parse_dates=[TIME_COLUMN_NAME]
                              )

    with open(PATH_TO_DROPPED_FEATURES_FILENAME) as fp:
        line = fp.readline()
        dropped_columns = set((line.rstrip()).split(","))

    for column_name in dropped_columns:
        holdout_matrix.drop(column_name, axis=1, inplace=True)

    timestamp2prediction = pd.DataFrame(columns=[TIME_COLUMN_NAME, TARGET_COLUMN_NAME])

    timestamp2prediction[TIME_COLUMN_NAME] = holdout_matrix[TIME_COLUMN_NAME]

    holdout_matrix.drop(TIME_COLUMN_NAME, axis=1, inplace=True)

    holdout_matrix.interpolate(method='linear', axis=0, limit_direction='both', inplace=True)

    return holdout_matrix, timestamp2prediction

########################################################################################################################


def _load_label2status():

    with open(PATH_TO_LABEL2STATUS_PKL, 'rb') as my_input:
        label2status = pickle.load(my_input)

    return label2status

########################################################################################################################


def execute():

    # ------------------------------------------------------------------------------------------------------------------

    label2status = _load_label2status()

    holdout_matrix, timestamp2prediction = _load_holdout_matrix()

    holdout_info = HoldoutDataClass(holdout_matrix, label2status, timestamp2prediction)

    # ------------------------------------------------------------------------------------------------------------------

    standard_scaler = _load_standard_scaler()

    model = _load_model()

    model_info = ModelClass(standard_scaler, model)

    # ------------------------------------------------------------------------------------------------------------------

    return holdout_info, model_info

########################################################################################################################
