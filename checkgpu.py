import os
import sys
import numpy as np
import lightgbm as lgb
import pyopencl as cl

print("===== 🔥 LightGBM GPU环境检查 =====")

# 检查1：验证Python环境和包版本
print("\n【1. 基础环境检查】")
print(f"Python版本: {sys.version}")
print(f"LightGBM版本: {lgb.__version__}")
print(f"pyopencl版本: {cl.__version__}")

# 检查2：验证OpenCL设备（核心！确认RTX4060是否被识别）
print("\n【2. OpenCL设备检查】")
try:
    platforms = cl.get_platforms()
    if not platforms:
        print("❌ 未检测到任何OpenCL平台，GPU训练无法启用！")
    else:
        gpu_available = False
        for platform_id, platform in enumerate(platforms):
            print(f"  平台ID {platform_id}: {platform.name} (版本: {platform.version})")
            devices = platform.get_devices()
            for device_id, device in enumerate(devices):
                device_type = cl.device_type.to_string(device.type)
                print(f"    设备ID {device_id}: {device.name} (类型: {device_type})")
                if "GPU" in device_type and "RTX 4060" in device.name:
                    gpu_available = True
                    target_platform_id = platform_id
                    target_device_id = device_id
        if gpu_available:
            print(f"✅ 检测到RTX4060！平台ID: {target_platform_id}, 设备ID: {target_device_id}")
        else:
            print("❌ 未检测到RTX4060 GPU设备，请检查显卡驱动！")
except Exception as e:
    print(f"❌ OpenCL检查失败: {str(e)}")

# 检查3：验证LightGBM是否支持GPU（运行最小测试用例）
print("\n【3. LightGBM GPU功能验证】")
try:
    # 生成测试数据
    X = np.random.randn(10000, 10)
    y = np.random.randn(10000)
    
    # 用GPU模式训练小模型
    params = {
        "device_type": "gpu",
        "gpu_platform_id": target_platform_id if 'target_platform_id' in locals() else 0,
        "gpu_device_id": target_device_id if 'target_device_id' in locals() else 0,
        "max_bin": 255,
        "gpu_use_dp": False,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1  # 关闭冗余输出
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    
    # 检查训练设备
    train_device = model.get_params()["device_type"]
    print(f"✅ LightGBM GPU训练成功！使用设备: {train_device}")
except Exception as e:
    print(f"❌ LightGBM GPU训练失败: {str(e)}")
    print("💡 常见原因：1. LightGBM未编译GPU支持；2. OpenCL设备ID错误；3. 显卡驱动缺失")

print("\n===== 📌 检查完成 =====")