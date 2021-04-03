import initialize
import perform_evaluation
import aggregate_models
from config import TESTSET
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#######################################################################################################################

print("\nSTART EVALUATING MODELS\n")

models_info, test_set_df, standard_scaler = initialize.execute()

print("\n###################################")
print("A) Scores per fold per algorithm")
print("###################################\n")

output = pd.DataFrame(columns=['Fold', 'Algorithm', 'BACC', 'ACCR'])

for model_name in models_info:

    experiments = models_info[model_name]

    for fold, experiment in experiments.items():

        model = initialize.load_model(experiment)

        eval_manager = experiment.eval_on_dataset.test_set

        if fold in standard_scaler:
            experiment_standard_scaler = standard_scaler[fold]
        else:
            experiment_standard_scaler = None

        output = perform_evaluation.execute(fold,
                                            eval_manager,
                                            model_name,
                                            model,
                                            test_set_df,
                                            TESTSET,
                                            experiment_standard_scaler,
                                            output)

print(output.to_string(index=False))

print("\n###################################")
print("B) Aggregated scores per algorithm")
print("###################################")

aggregate_models.execute(models_info, test_set_df)

print("\nEND EVALUATING MODELS\n")
