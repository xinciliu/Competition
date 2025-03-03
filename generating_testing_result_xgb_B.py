import xgboost as xgb
import os
import joblib

# 加载模型
def prediction_model(model, feature_X):
    feature_X = feature_X.drop(columns=['LogTime'])
    dtest = xgb.DMatrix(feature_X)
    predictions = model.predict(dtest)
    return predictions

###跑测试-加油
filepathin = "./select_type_B_feature/" ##本地修改
filepathout = "./select_type_B_feature_result_xgb6/"
import os
import pandas as pd
all_items = os.listdir(filepathin)
# 筛选出文件名称
file_names = [item for item in all_items if os.path.isfile(os.path.join(filepathin, item))]
exits_items = os.listdir(filepathout)
exits_name_list = [item for item in exits_items if os.path.isfile(os.path.join(filepathout, item))]
#model_path
model_path = "xgb_b_large_224.model"
model = xgb.Booster()
model.load_model(model_path)

model_path2 = "xgb_b_small_224.model"
model_2 = xgb.Booster()
model_2.load_model(model_path2)
for filename in file_names:
    if filename in exits_name_list:
        print(filename)
        continue
    filepath = filepathin + filename
    feature_df = pd.read_csv(filepath)
    ##预测数据
    rows = feature_df.shape[0]
    if rows < 30:
        model_used = model_2
    else:
        model_used = model
    predictions = prediction_model(model_used, feature_df)
    #把其中标注为1的数据拿出来，然后打印为csv
    total_predict = list(predictions)
    feature_df["predict_result"] = total_predict
    fileout = filepathout + filename
    feature_df.to_csv(fileout, index=False)

