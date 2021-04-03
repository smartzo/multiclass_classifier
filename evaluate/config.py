import os

OPTUNA_DUMP = 'optuna_dump'
TEST_SET_FILENAME = 'test_set.csv'
STANDARD_SCALER = 'StandardScaler'
IS_DEMO_VERSION = True
XGBOOST = 'xgboost'
PYTORCH = 'pytorch'
TARGET_COLUMN_NAME = 'status'
TIME_COLUMN_NAME = 'timestamp'
BEST_TRIAL = 'best_trial'
FEATURES = 'features'
TARGET = 'target'

########################################################################################################################

if os.name == 'nt':
    PATH_TO_BUILD = r"..\build"
else:
    PATH_TO_BUILD = r"../build"

########################################################################################################################

PATH_TO_OPTUNA_DUMP = os.path.join(PATH_TO_BUILD, OPTUNA_DUMP)
PATH_TO_TEST_SET = os.path.join(PATH_TO_BUILD, TEST_SET_FILENAME)
PATH_TO_STANDARD_SCALER = os.path.join(PATH_TO_BUILD, STANDARD_SCALER)

########################################################################################################################

APPLY_STANDARD_SCALER = {XGBOOST: False,
                         PYTORCH: True}

########################################################################################################################

PREDICTIONS = 'predictions'
TRAINSET = 'trainset'
TUNESET = 'tuneset'
TESTSET = 'testset'

########################################################################################################################

AGGREGATED_PREDICTIONS = 'aggregated_predictions'
COMBINED_ALGORITHMS = 'combined_algorithms'
BEST_PER_ALGORITHM = 'per_algorithm'
ALL_COMBINATIONS = 'all'
BEST_COMBINATION = 'best'
MODEL_DELIMITER = '_'

########################################################################################################################

if IS_DEMO_VERSION:

    BACC_THRESHOLDS = {XGBOOST: 0.0,
                       PYTORCH: 0.0}
else:

    BACC_THRESHOLDS = {XGBOOST: 78.0,
                       PYTORCH: 85.0}

########################################################################################################################
