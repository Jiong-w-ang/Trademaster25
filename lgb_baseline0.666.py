import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

# 1. 补全WMAE函数（必须有）
def wmae(mae_s, mae_m, mae_l):
    return (0.5 * mae_s + 0.3 * mae_m + 0.2 * mae_l) * 100

# 2. 读取数据（确认文件名和列名匹配）
train = pd.read_csv("../data/train_v2.csv")
test  = pd.read_csv("../data/test_v2.csv")

feat_cols = [c for c in train.columns if c.startswith("feature_")]

# 时间排序（确认数据中有date_id/minute_id列）
train = train.sort_values(["date_id","minute_id"]).reset_index(drop=True)
test  = test.sort_values(["date_id","minute_id"]).reset_index(drop=True)

# 按date_id切分验证集
dates = np.array(sorted(train["date_id"].unique()))
split = int(len(dates)*0.9)
train_dates = dates[:split]
valid_dates = dates[split:]

GAP_DAYS = 2
if len(train_dates) > GAP_DAYS:
    train_dates = train_dates[:-GAP_DAYS]

tr = train[train["date_id"].isin(train_dates)].copy()
va = train[train["date_id"].isin(valid_dates)].copy()

# 收缩函数（优化预测值）
def best_shrink(y, p, k_max=8.0, steps=801):
    ks = np.linspace(0.0, k_max, steps)
    # 形状：(n, steps)
    errs = np.abs(y[:, None] - (p[:, None] * ks[None, :]))
    maes = errs.mean(axis=0)
    return float(ks[int(np.argmin(maes))])


# ========== 核心修复：统一函数名+匹配返回值 ==========
# 方案1：函数返回4个值（匹配调用时的4个接收值）
def fit_one(target_col, seed=42):
    print(f"开始训练 {target_col} 模型...")
    model = LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.03,
        num_leaves=32,
        min_data_in_leaf=300,  # 保留该参数，Warning无影响
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression_l1",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(
        tr[feat_cols], tr[target_col],
        eval_set=[(va[feat_cols], va[target_col])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(200)]
    )

    p_va = model.predict(va[feat_cols])
    p_te = model.predict(test[feat_cols])

    # 收缩优化
    k = best_shrink(va[target_col].values, p_va)
    p_va, p_te = k*p_va, k*p_te

    # 裁剪极端值
    lo, hi = np.quantile(tr[target_col].values, [0.001, 0.999])
    p_va = np.clip(p_va, lo, hi)
    p_te = np.clip(p_te, lo, hi)

    # 修复：返回4个值（第一个值用None占位，匹配调用格式）
    return None, k, p_va, p_te

# ========== 训练三个目标（调用格式不变） ==========
_, k_s, pred_va_s, pred_te_s = fit_one("target_short")
_, k_m, pred_va_m, pred_te_m = fit_one("target_medium")
_, k_l, pred_va_l, pred_te_l = fit_one("target_long")

# ========== 评估+导出 ==========
mae_s = np.mean(np.abs(pred_va_s - va["target_short"].values))
mae_m = np.mean(np.abs(pred_va_m - va["target_medium"].values))
mae_l = np.mean(np.abs(pred_va_l - va["target_long"].values))
score = wmae(mae_s, mae_m, mae_l)
print("===============================================")
print("Shrink k (short/med/long):", k_s, k_m, k_l)
print("MAE short:", mae_s)
print("MAE medium:", mae_m)
print("MAE long:", mae_l)
print("WMAE:", score)

# 生成提交文件
sub = pd.DataFrame({
    "id": test["id"].values,
    "target_short": pred_te_s,
    "target_medium": pred_te_m,
    "target_long": pred_te_l
})
sub_path = "../outputs/submission_lgb_es_shrink.csv"
sub.to_csv(sub_path, index=False)
print("Saved:", sub_path)