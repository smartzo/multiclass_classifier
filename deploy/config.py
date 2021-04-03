import os

STANDARD_SCALER = 'StandardScaler'
LABEL2STATUS_PKL = "label2status.pkl"
DROPPED_FEATURES_FILENAME = "dropped_features.csv"
IS_DEMO_VERSION = True
AGGREGATED_PREDICTIONS = 'aggregated_predictions'
COMBINED_ALGORITHMS = 'combined_algorithms'
BEST_COMBINATION = 'best'
MODEL_DELIMITER = "_"
FEATURES = 'features'
TARGET = 'target'
TIME_COLUMN_NAME = 'timestamp'
TARGET_COLUMN_NAME = 'status'
XGBOOST = 'xgboost'
PYTORCH = 'pytorch'

APPLY_STANDARD_SCALER = {XGBOOST: False,
                         PYTORCH: True}

########################################################################################################################

if os.name == 'nt':
    PATH_TO_BUILD = r"..\build"
    PATH_TO_EVALUATE = r"..\evaluate"
else:
    PATH_TO_BUILD = r"../build"
    PATH_TO_EVALUATE = r"../evaluate"

########################################################################################################################

PATH_TO_HOLDOUT_SET = "holdout_set.csv"

########################################################################################################################

PATH_TO_LABEL2STATUS_PKL = os.path.join(PATH_TO_BUILD, LABEL2STATUS_PKL)

PATH_TO_DROPPED_FEATURES_FILENAME = os.path.join(PATH_TO_BUILD, DROPPED_FEATURES_FILENAME)

PATH_TO_STANDARD_SCALER = os.path.join(PATH_TO_BUILD, STANDARD_SCALER)

PATH_TO_BEST_MODEL = os.path.join(PATH_TO_EVALUATE,
                                  AGGREGATED_PREDICTIONS,
                                  COMBINED_ALGORITHMS,
                                  BEST_COMBINATION)

########################################################################################################################

if IS_DEMO_VERSION:
    pass

########################################################################################################################
