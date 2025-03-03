import pandas as pd
import sys
import os

def training_datasets(filepathin, ticket_csv):
    """
    """
    #读ticket_df
    ticket_df = pd.read_csv(ticket_csv)
    result_dict = ticket_df.set_index('sn_name').to_dict(orient='index')
    train_df_lis = []
    train_df_lis_small = []
    all_items = os.listdir(filepathin)
    file_names = [item for item in all_items if os.path.isfile(os.path.join(filepathin, item))]
    for sn_name in all_items:
        filp = filepathin + sn_name
        sn_name = sn_name.replace(".feather", "")
        feature_df = pd.read_csv(filp)
        if sn_name in result_dict:
            alarm_time = result_dict[sn_name]["alarm_time"]
            ###所有<= alarm time >= alarmtime - 30 tian的都定义成1
            #只考虑alarm_time之前的数据, label = 1
            alarm_window = alarm_time - 3*24*3600
            train_df = feature_df[(feature_df["LogTime"] <= alarm_time) & (feature_df["LogTime"] >= alarm_window)]
            #这部分是1
            if train_df.empty:
                continue
            train_df['label'] = 1
        else:
            train_end_date = 1717171200
            end_time = train_end_date
            start_time = train_end_date - 60*24*3600
            train_df = feature_df[(feature_df["LogTime"] <= end_time) & (feature_df["LogTime"] >= start_time)]
            if train_df.empty:
                continue
            train_df['label'] = 0
        rows = train_df.shape[0]
        if rows < 30:
            train_df_lis_small.append(train_df)
        else:
            train_df_lis.append(train_df)
    combined_df = pd.concat(train_df_lis, ignore_index=True)
    combined_df_small = pd.concat(train_df_lis_small, ignore_index=True)
    return combined_df, combined_df_small

if __name__ == "__main__":
    df_dict = {}
    filepathin = sys.argv[1]
    ticket_csv = sys.argv[2]
    ###跑完之后进行训练
    training_data_out = sys.argv[3]
    small_training_out = sys.argv[4]
    ##generating_dataframe
    total_training_dataframe, small_total_training_dataframe = training_datasets(filepathin, ticket_csv)
    #total_training_dataframe = total_training_dataframe.drop(columns=['LogTime'])
    total_training_dataframe.to_csv(training_data_out, index=False)
    small_total_training_dataframe.to_csv(small_training_out, index=False)
