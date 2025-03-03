# Competition
Here is the source code for competition

# 1. feature engineering process
generating_training_features.py -> this is same as the baseline

# 2. generating training data
generating_training_data.py -> generate two different training datasize for both larger datasize and smaller datasize. Use 3 days before failure as positive label, and all other points that without failure as negative label.

# 3. generating xgb model
Use the result from step 2 to generate different xgb models for both large and small datasize.

# 4. generating prediction result
generating_testing_result_xgb_A.py
generating_testing_result_xgb_B.py

generate prediction result for type A and B with the model from test 3.

# 5. generate submit result file
generate_submit_A_xgb_new.py
generate_submit_B_xgb_new.py

generate submit files for type A and B.

