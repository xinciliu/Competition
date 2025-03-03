import os
import pandas as pd
import sys

horizon = sys.argv[1]
timestamp1 = 1717171200
timestamp2 = 1722441600
###生成一个submission_file
filepath = "./select_type_A_feature_result_xgb3/"
all_items = os.listdir(filepath)
# 筛选出文件名称
file_names = [item for item in all_items if os.path.isfile(os.path.join(filepath, item))]
df_total_list = []
for filename in file_names:
    fil = filepath + filename
    df = pd.read_csv(fil)
    filtered_df = df[(df['LogTime'] >= timestamp1) & (df['LogTime'] < timestamp2)]
    filtered_df = filtered_df[filtered_df['predict_result'] > float(horizon)]
    selected_df = filtered_df[['LogTime']]
    selected_df["sn_name"] = filename.replace(".feather", "")
    selected_df["serial_number_type"] = "A"
    df_total_list.append(selected_df)
result_row = pd.concat(df_total_list, axis=0, ignore_index=True)
filtered_df = result_row.reindex(columns=["sn_name", "LogTime", "serial_number_type"])
new_name = ["sn_name", "prediction_timestamp", "serial_number_type"]
filtered_df.columns = new_name
filtered_df.to_csv("submission_A_225_new_baseline_xgb" + horizon.replace(".", "") + ".csv", index=False)
