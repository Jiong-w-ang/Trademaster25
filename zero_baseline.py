import pandas as pd
import numpy as np

# 定义WMAE计算函数（和比赛评分一致）
def wmae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# 读取训练数据（路径对应你的data文件夹）
train = pd.read_csv("../data/train.csv")
print("数据列名：", train.columns.tolist())
print("数据总行数：", len(train))

# 替代方案：用id列做切分（无date_id的情况下）
# 取最后10%的数据做验证集（模拟时间切分的逻辑）
split = int(len(train) * 0.9)
tr = train.iloc[:split]  # 训练集：前90%
va = train.iloc[split:]  # 验证集：后10%

# 零预测：所有目标都预测0
zero_pred_s = np.zeros(len(va))
zero_pred_m = np.zeros(len(va))
zero_pred_l = np.zeros(len(va))

# 计算各目标MAE和WMAE
mae_s = wmae(va["target_short"].values, zero_pred_s)
mae_m = wmae(va["target_medium"].values, zero_pred_m)
mae_l = wmae(va["target_long"].values, zero_pred_l)
wmae_score = (0.5*mae_s + 0.3*mae_m + 0.2*mae_l) * 100

# 打印结果（记录这个分数，作为最低基准）
print("\n===== 零预测基线结果 =====")
print(f"验证集样本数：{len(va)}")
print(f"MAE short: {mae_s:.6f}")
print(f"MAE medium: {mae_m:.6f}")
print(f"MAE long: {mae_l:.6f}")
print(f"最终 WMAE 分数: {wmae_score:.6f}")