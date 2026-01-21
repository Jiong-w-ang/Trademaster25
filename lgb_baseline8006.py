import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

# --------------------------
# Config
# --------------------------
N_TAIL_FOLDS = 3          # 只用末尾3折
N_BLOCKS = 6              # 把末尾日期切成6个连续block，再取最后3个block做valid
GAP_DAYS = 2
SEEDS = [2024, 2025, 2026]  # 新增：3个基础seed
# 新增：不同target的alpha搜索范围
ALPHA_RANGES = {
    "target_short": np.linspace(0.0, 1.1, 441),  # 允许轻微放大
    "target_medium": np.linspace(0.0, 0.8, 321), # 更保守的收缩范围
    "target_long": np.linspace(0.0, 0.9, 361)    # 适度收缩
}
ALPHAS = np.linspace(0.0, 1.0, 401)  # 兜底alpha范围

def wmae(mae_s, mae_m, mae_l):
    return (0.5*mae_s + 0.3*mae_m + 0.2*mae_l) * 100

def l1_calibrate(y, p, alphas=ALPHAS):
    """
    MAE 下的线性校准：pred = a*p + b
    对每个 a，最优 b 是 median(y - a*p)
    返回最优 (a,b)
    """
    best = (1e18, 0.0, 0.0)
    for a in alphas:
        r = y - a*p
        b = np.median(r)
        mae = np.mean(np.abs(r - b))
        if mae < best[0]:
            best = (mae, float(a), float(b))
    return best[1], best[2]  # a, b

def fit_lgb(X_tr, y_tr, X_va, y_va, seed=42):
    model = LGBMRegressor(
        n_estimators=20000,
        learning_rate=0.03,
        num_leaves=32,
        min_data_in_leaf=300,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression_l1",
        random_state=seed,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)]
    )
    return model

# --------------------------
# Load
# --------------------------
train = pd.read_csv("../data/train_v2.csv")
test  = pd.read_csv("../data/test_v2.csv")

feat_cols = [c for c in train.columns if c.startswith("feature_")]

train = train.sort_values(["date_id","minute_id"]).reset_index(drop=True)
test  = test.sort_values(["date_id","minute_id"]).reset_index(drop=True)

dates = np.array(sorted(train["date_id"].unique()))
tail_dates = dates[-int(len(dates)*0.3):]  # 只取最后30%的日期做“近端CV池”，更贴近test

# 把 tail_dates 分成 N_BLOCKS 个连续块
blocks = np.array_split(tail_dates, N_BLOCKS)
valid_blocks = blocks[-N_TAIL_FOLDS:]  # 取末尾3个块作为valid折

targets = ["target_short","target_medium","target_long"]

# 初始化全局容器：存储所有seed的预测结果和校准参数
test_preds_all = {t: [] for t in targets}
cal_params_all = {t: [] for t in targets}

# 外层循环：遍历3个seed
for seed0 in SEEDS:
    print(f"\n========== Processing Seed: {seed0} ==========")
    # 每个seed内部的3折容器
    test_preds = {t: [] for t in targets}
    cal_params = {t: [] for t in targets}

    for fi, v_dates in enumerate(valid_blocks):
        v_dates = np.array(v_dates)
        v_start = v_dates.min()

        # 训练日期：所有 < valid起点 的日期（再做gap）
        tr_dates = dates[dates < v_start]
        if len(tr_dates) > GAP_DAYS:
            tr_dates = tr_dates[:-GAP_DAYS]

        tr_idx = train["date_id"].isin(tr_dates).values
        va_idx = train["date_id"].isin(v_dates).values

        X_tr = train.loc[tr_idx, feat_cols]
        X_va = train.loc[va_idx, feat_cols]
        X_te = test[feat_cols]

        print(f"Fold {fi}: tr_dates={len(tr_dates)}, va_dates={len(v_dates)}, tr_rows={tr_idx.sum()}, va_rows={va_idx.sum()}")

        for t in targets:
            y_tr = train.loc[tr_idx, t].values
            y_va = train.loc[va_idx, t].values

            # 关键：每个seed+fold用不同的随机种子
            model = fit_lgb(X_tr, y_tr, X_va, y_va, seed=seed0 + fi)

            p_va = model.predict(X_va)
            p_te = model.predict(X_te)

            # 不同target使用定制的alpha范围
            a, b = l1_calibrate(y_va, p_va, alphas=ALPHA_RANGES[t])
            p_te_cal = a * p_te + b

            # clip（用训练分布）
            lo, hi = np.quantile(y_tr, [0.001, 0.999])
            p_te_cal = np.clip(p_te_cal, lo, hi)

            test_preds[t].append(p_te_cal)
            cal_params[t].append((a,b))
    
    # 计算当前seed的3折平均预测
    seed_pred = {t: np.mean(np.vstack(test_preds[t]), axis=0) for t in targets}
    # 存入全局容器
    for t in targets:
        test_preds_all[t].append(seed_pred[t])
        cal_params_all[t].extend(cal_params[t])
    
    # 打印当前seed的校准参数均值
    print(f"\nSeed {seed0} 校准参数均值：")
    for t in targets:
        a_mean = np.mean([x[0] for x in cal_params[t]])
        b_mean = np.mean([x[1] for x in cal_params[t]])
        print(f"{t}: mean alpha={a_mean:.3f}, mean bias={b_mean:.6g}")

# 计算所有seed的平均预测（最终提交用）
final_test_pred = {t: np.mean(np.vstack(test_preds_all[t]), axis=0) for t in targets}

# 输出所有seed合并后的校准参数均值（用于反馈）
print("\n========== 所有Seed合并后的校准参数均值 ==========")
for t in targets:
    a_mean = np.mean([x[0] for x in cal_params_all[t]])
    b_mean = np.mean([x[1] for x in cal_params_all[t]])
    print(f"{t}: mean alpha={a_mean:.3f}, mean bias={b_mean:.6g}")

# 生成提交文件
sub = pd.DataFrame({
    "id": test["id"].values,
    "target_short": final_test_pred["target_short"],
    "target_medium": final_test_pred["target_medium"],
    "target_long": final_test_pred["target_long"]
})
sub_path = "../outputs/submission_lgb_tailcv_3seed_bagging.csv"
sub.to_csv(sub_path, index=False)
print("\nSaved:", sub_path)