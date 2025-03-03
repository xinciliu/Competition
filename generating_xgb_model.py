import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import pandas as pd

def xgb_generating_model(training_df):
    """
    训练 XGBoost 模型
    :param training_df: 包含特征和标签的 DataFrame
    :return: 训练好的 XGBoost 模型
    """
    # 分离特征 (X) 和标签 (y)
    X = training_df.drop(columns=['label'])  # 所有列除了 label 列作为特征
    y = training_df['label']  # label 列作为目标变量
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    # 将数据转换为 DMatrix 格式（XGBoost 的高效数据格式）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    # 设置 XGBoost 参数
    params = {
        'objective': 'binary:logistic',  # 二分类问题
        'eval_metric': 'logloss',        # 评估指标
        'max_depth': 6,                  # 树的最大深度
        'eta': 0.1,                      # 学习率
        'subsample': 0.8,                # 每棵树使用的样本比例
        'colsample_bytree': 0.8,         # 每棵树使用的特征比例
        'seed': 42                       # 随机种子
    }
    # 训练模型
    num_rounds = 800  # 迭代次数
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dval, 'eval')],  # 在验证集上评估
        early_stopping_rounds=10,  # 早停，防止过拟合
        verbose_eval=10  # 每 10 轮打印一次评估结果
    )
    # 在验证集上评估模型
    y_pred = model.predict(dval)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]  # 将概率转换为类别
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return model

if __name__ == "__main__":
    fi = sys.argv[1]
    model_name = sys.argv[2]
    training_df = pd.read_csv(fi)
    training_df = training_df.drop(columns=['LogTime'])
    model = xgb_generating_model(training_df)
    model.save_model(model_name)
