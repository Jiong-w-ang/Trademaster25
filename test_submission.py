import pandas as pd
# 1. 加载训练集（替换为你的实际路径）
train = pd.read_csv("../data/train_v2.csv")
# 2. 加载你那版有问题的预测结果（替换为你的预测文件路径）
preds = pd.read_csv("../outputs/submission_lgb_param_bagging.csv")

# 3. 对比分布
target_cols = ["target_short","target_medium","target_long"]
for col in target_cols:
    print(f"\n===== {col} =====")
    print("训练集分布：")
    print(train[col].describe(percentiles=[0.001,0.01,0.5,0.99,0.999]))
    print("预测分布：")
    print(preds[col].describe(percentiles=[0.001,0.01,0.5,0.99,0.999]))
    # 计算均值偏移率
    mean_diff = abs(preds[col].mean() - train[col].mean()) / abs(train[col].mean())
    print(f"均值偏移率：{mean_diff:.4f}（>0.1就是偏移严重）")