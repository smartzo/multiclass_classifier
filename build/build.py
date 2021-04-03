import manage_data
import manage_optuna
from config import N_FOLDS
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################################

print("\nSTART BUILDING MODELS\n")

optuna_models = manage_optuna.make_optuna_dir()

input_data = manage_data.get_input_data()

manage_data.get_test(input_data)

for fold in range(N_FOLDS):

    ml_data = manage_data.get_train_and_tune(input_data, fold)

    for optuna_model in optuna_models:

        optuna_model_dir = optuna_model.directory

        ml_algorithm_optuna = optuna_model.ml_algorithm_script

        ml_algorithm_optuna.execute(input_data, ml_data, optuna_model_dir, fold)

print("\nEND BUILDING MODELS\n")
