import load_data
import predict

########################################################################################################################

print("\nSTART DEPLOYMENT EXAMPLE\n")

holdout_info, model_info = load_data.execute()

predict.execute(holdout_info, model_info)

print(holdout_info.timestamp2prediction)

print("\nEND DEPLOYMENT EXAMPLE\n")

########################################################################################################################
