import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor

os.makedirs("../outputs", exist_ok=True)

# =========================
# GPU config (RTX4060 Laptop)
# =========================
GPU_KW = dict(
    device_type="gpu",
    gpu_platform_id=0,
    gpu_device_id=0,
    max_bin=255,
    gpu_use_dp=False
)

# =========================
# Competition / CV config
# =========================
SEEDS = [2024, 2025, 2026]

TAIL_RATIO = 0.30
N_BLOCKS = 6
N_TAIL_FOLDS = 3
GAP_DAYS = 2

TARGETS = ["target_short", "target_medium", "target_long"]

PARAM_SETS = {
    "base": dict(
        num_leaves=32, min_data_in_leaf=300,
        reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.03
    ),
    "cons": dict(
        num_leaves=24, min_data_in_leaf=600,
        reg_alpha=0.2, reg_lambda=2.0,
        subsample=0.75, colsample_bytree=0.75,
        learning_rate=0.03
    ),
    "expr": dict(
        num_leaves=48, min_data_in_leaf=200,
        reg_alpha=0.05, reg_lambda=0.8,
        subsample=0.85, colsample_bytree=0.85,
        learning_rate=0.03
    ),
}
PARAM_SETS_BY_T = {
    "target_short": ["base", "cons"],        # short 更稳
    "target_medium": ["base", "cons"],       # medium 禁用 expr
    "target_long": ["base", "expr"],         # long 保留 expr
}
EARLY_STOP = 300

# 关键：short 更严格，避免放大导致 Public 翻车
ALPHA_GRID = {
    "target_short":  np.linspace(0.0, 1.02, 409),
    "target_medium": np.linspace(0.05, 0.70, 326),
    "target_long":   np.linspace(0.05, 0.85, 341),
}

# 关键：权重约束，防止出现“某个权重为0”的极端 fold-overfit
# 强制 base 至少占比一定比例（你可以激进调小/调大测试）
W_BOUNDS = {
    "target_short":  (0.30, 1.00),  # base最低30%
    "target_medium": (0.50, 1.00),  # medium最容易过拟合：base最低50%
    "target_long":   (0.40, 1.00),
}

WEIGHT_STEP = 0.05  # 网格步长；0.02更细但更慢


def wmae(mae_s, mae_m, mae_l):
    return (0.5 * mae_s + 0.3 * mae_m + 0.2 * mae_l) * 100


def l1_calibrate(y, p, alphas):
    """MAE/L1 校准：pred = a*p + b；给定a，最优b=median(y-a*p)"""
    best_mae = 1e18
    best_a, best_b = 0.0, 0.0
    for a in alphas:
        r = y - a * p
        b = np.median(r)
        mae = np.mean(np.abs(r - b))
        if mae < best_mae:
            best_mae = mae
            best_a, best_b = float(a), float(b)
    return best_a, best_b


def fit_lgb(X_tr, y_tr, X_va, y_va, seed, params):
    model = LGBMRegressor(
        n_estimators=20000,
        objective="regression_l1",
        random_state=seed,
        n_jobs=-1,
        **GPU_KW,
        **params
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(EARLY_STOP), lgb.log_evaluation(0)]
    )
    return model


def make_tail_valid_blocks(dates):
    tail_dates = dates[-int(len(dates) * TAIL_RATIO):]
    blocks = np.array_split(tail_dates, N_BLOCKS)
    return blocks[-N_TAIL_FOLDS:]


def sanity_check_submission(sub, test_ids):
    pred_cols = ["target_short", "target_medium", "target_long"]
    assert list(sub.columns) == ["id"] + pred_cols, f"Bad columns: {sub.columns}"
    assert len(sub) == len(test_ids), f"Row mismatch: sub={len(sub)}, test={len(test_ids)}"
    assert sub["id"].nunique() == len(sub), "Duplicate ids in submission"
    missing = len(set(test_ids) - set(sub["id"]))
    extra = len(set(sub["id"]) - set(test_ids))
    assert missing == 0 and extra == 0, f"ID mismatch: missing={missing}, extra={extra}"
    assert not sub[pred_cols].isna().any().any(), "NaN found in predictions"
    assert not np.isinf(sub[pred_cols].values).any(), "inf found in predictions"


def best_global_weights_3_mae(y_all, p_base_all, p_cons_all, p_expr_all, step, wb_min):
    """
    在 w_base>=wb_min, w>=0, sum=1 的约束下，网格搜索最小 MAE 的全局权重。
    只搜索 (w_base, w_cons)，w_expr=1-wb-wc。
    """
    best_mae = 1e18
    best = (1.0, 0.0, 0.0)

    grid = np.arange(0.0, 1.0 + 1e-12, step)
    for wb in grid:
        if wb < wb_min - 1e-12:
            continue
        for wc in grid:
            we = 1.0 - wb - wc
            if we < -1e-12:
                continue
            if we < 0:
                we = 0.0
            p = wb * p_base_all + wc * p_cons_all + we * p_expr_all
            mae = np.mean(np.abs(y_all - p))
            if mae < best_mae:
                best_mae = mae
                best = (float(wb), float(wc), float(we))
    return best


# =========================
# Load
# =========================
train = pd.read_csv("../data/train_v2.csv")
test = pd.read_csv("../data/test_v2.csv")

feat_cols = [c for c in train.columns if c.startswith("feature_")]

# safe cleaning
train[feat_cols] = train[feat_cols].replace([np.inf, -np.inf], np.nan)
test[feat_cols] = test[feat_cols].replace([np.inf, -np.inf], np.nan)
med = train[feat_cols].median()
train[feat_cols] = train[feat_cols].fillna(med)
test[feat_cols] = test[feat_cols].fillna(med)

train = train.sort_values(["date_id", "minute_id"]).reset_index(drop=True)
test = test.sort_values(["date_id", "minute_id"]).reset_index(drop=True)

dates = np.array(sorted(train["date_id"].unique()))
valid_blocks = make_tail_valid_blocks(dates)
X_test = test[feat_cols]

# =========================
# Phase 1: run folds, collect per-fold (paramset-averaged) predictions
# =========================
# For each fold & target: we will store paramset-averaged preds on valid and test
fold_store = []  # list of dict per fold: {t: {base/cons/expr: (p_va, p_te), y_va, y_tr}}
print("Valid blocks:", [ (int(np.min(b)), int(np.max(b)), len(b)) for b in valid_blocks ])

for fi, v_dates in enumerate(valid_blocks):
    v_dates = np.array(v_dates)
    v_start = v_dates.min()

    tr_dates = dates[dates < v_start]
    if len(tr_dates) > GAP_DAYS:
        tr_dates = tr_dates[:-GAP_DAYS]

    tr_idx = train["date_id"].isin(tr_dates).values
    va_idx = train["date_id"].isin(v_dates).values

    X_tr = train.loc[tr_idx, feat_cols]
    X_va = train.loc[va_idx, feat_cols]

    fold_info = {"fi": fi, "tr_idx": tr_idx, "va_idx": va_idx, "per_target": {}}

    print(f"\n[Phase1] Fold {fi}: tr_rows={tr_idx.sum()}, va_rows={va_idx.sum()}, tr_dates={len(tr_dates)}, va_dates={len(v_dates)}")

    for t in TARGETS:
        y_tr = train.loc[tr_idx, t].values
        y_va = train.loc[va_idx, t].values

        per_set = {}
        for pname, pset in PARAM_SETS.items():
            va_list, te_list = [], []
            for seed0 in SEEDS:
                model = fit_lgb(X_tr, y_tr, X_va, y_va, seed=seed0 + 10 * fi, params=pset)
                va_list.append(model.predict(X_va))
                te_list.append(model.predict(X_test))
            per_set[pname] = (np.mean(np.vstack(va_list), axis=0),
                              np.mean(np.vstack(te_list), axis=0))

        fold_info["per_target"][t] = {
            "y_tr": y_tr, "y_va": y_va,
            "p_va_base": per_set["base"][0], "p_te_base": per_set["base"][1],
            "p_va_cons": per_set["cons"][0], "p_te_cons": per_set["cons"][1],
            "p_va_expr": per_set["expr"][0], "p_te_expr": per_set["expr"][1],
        }

    fold_store.append(fold_info)

# =========================
# Phase 2: learn GLOBAL weights per target by concatenating all fold-valid
# =========================
global_weights = {}
for t in TARGETS:
    y_all, p_b_all, p_c_all, p_e_all = [], [], [], []
    for fd in fold_store:
        d = fd["per_target"][t]
        y_all.append(d["y_va"])
        p_b_all.append(d["p_va_base"])
        p_c_all.append(d["p_va_cons"])
        p_e_all.append(d["p_va_expr"])
    y_all = np.concatenate(y_all)
    p_b_all = np.concatenate(p_b_all)
    p_c_all = np.concatenate(p_c_all)
    p_e_all = np.concatenate(p_e_all)

    wb_min, _ = W_BOUNDS[t]
    wb, wc, we = best_global_weights_3_mae(y_all, p_b_all, p_c_all, p_e_all, step=WEIGHT_STEP, wb_min=wb_min)
    global_weights[t] = (wb, wc, we)
    print(f"\n[Global W] {t}: w(base,cons,expr)=({wb:.3f},{wc:.3f},{we:.3f})  (base>= {wb_min:.2f})")

# =========================
# Phase 3: apply GLOBAL weights per fold, then fold-specific calibration + clip
# =========================
fold_test_preds = {t: [] for t in TARGETS}
fold_metrics = []
calib_collect = {t: [] for t in TARGETS}

for fd in fold_store:
    fi = fd["fi"]
    maes = {}

    print(f"\n[Phase3] Fold {fi}")

    for t in TARGETS:
        d = fd["per_target"][t]
        y_tr, y_va = d["y_tr"], d["y_va"]

        wb, wc, we = global_weights[t]

        p_va_ens = wb*d["p_va_base"] + wc*d["p_va_cons"] + we*d["p_va_expr"]
        p_te_ens = wb*d["p_te_base"] + wc*d["p_te_cons"] + we*d["p_te_expr"]

        # fold-specific calibration on valid
        a, b = l1_calibrate(y_va, p_va_ens, ALPHA_GRID[t])
        p_va_cal = a*p_va_ens + b
        p_te_cal = a*p_te_ens + b

        # clip by fold train distribution
        lo, hi = np.quantile(y_tr, [0.001, 0.999])
        p_va_cal = np.clip(p_va_cal, lo, hi)
        p_te_cal = np.clip(p_te_cal, lo, hi)

        mae = float(np.mean(np.abs(y_va - p_va_cal)))
        maes[t] = mae
        fold_test_preds[t].append(p_te_cal)
        calib_collect[t].append((a, b))

        print(f"  {t}: MAE={mae:.6f} | alpha={a:.3f} bias={b:.6g}")

    fw = wmae(maes["target_short"], maes["target_medium"], maes["target_long"])
    fold_metrics.append(fw)
    print(f"Fold {fi} WMAE: {fw:.6f}")

print("\n===== Tail-CV Summary =====")
print("WMAE mean:", float(np.mean(fold_metrics)), "std:", float(np.std(fold_metrics)))

for t in TARGETS:
    a_mean = float(np.mean([x[0] for x in calib_collect[t]]))
    b_mean = float(np.mean([x[1] for x in calib_collect[t]]))
    wb, wc, we = global_weights[t]
    print(f"{t}: global w=({wb:.3f},{wc:.3f},{we:.3f}) | mean alpha={a_mean:.3f}, mean bias={b_mean:.6g}")

# =========================
# Final submission (average across folds)
# =========================
final_pred = {t: np.mean(np.vstack(fold_test_preds[t]), axis=0) for t in TARGETS}

sub = pd.DataFrame({
    "id": test["id"].values,
    "target_short": final_pred["target_short"],
    "target_medium": final_pred["target_medium"],
    "target_long": final_pred["target_long"],
})

sanity_check_submission(sub, test["id"].values)

out_path = "../outputs/submission_lgb_gpu_tailcv_globalw_parambag.csv"
sub.to_csv(out_path, index=False)
print("\nSaved:", out_path)

print(sub[["target_short", "target_medium", "target_long"]].describe(percentiles=[0.001, 0.01, 0.5, 0.99, 0.999]))
