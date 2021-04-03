Execution order:

1) Script for building the classifier:
   build\build.py
   Model construction is set to "demo version" for quick execution (~3 mins).
   Constructs:
   * Directories: "optuna_dump", "StandardScaler"
   * Files: "dropped_features.csv", "test_set.csv", "label2status.pkl"

2) Script for evaluating the classifier:
   evaluate\evaluate.py
   Execution time is <1min.
   Constructs:
   * Directories: "aggregated_predictions", "predictions"
   * Files: six .jpg figures

3) Demonstration of how the classifier is deployed in practice:
   deploy\deploy.py
   Execution time <1min.